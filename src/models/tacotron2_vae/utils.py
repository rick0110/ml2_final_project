import torch
from typing import List, Tuple, Any


def get_mask_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    """
    Generates a boolean mask from a tensor of lengths.

    This function is crucial for handling variable-length sequences in a batch,
    preventing attention mechanisms or other operations from attending to padding tokens.

    Args:
        lengths (torch.Tensor): A 1D tensor of integers representing the lengths
                                of sequences in a batch. Shape: (batch_size,).

    Returns:
        torch.Tensor: A boolean mask tensor. `True` indicates a valid position,
                      `False` indicates a padding position.
                      Shape: (batch_size, max_len), where max_len is the maximum
                      length in the `lengths` tensor.

    Examples:
        >>> lengths = torch.tensor([3, 5, 2])
        >>> mask = get_mask_from_lengths(lengths)
        >>> print(mask)
        tensor([[ True,  True,  True, False, False],
                [ True,  True,  True,  True,  True],
                [ True,  True, False, False, False]])
    """
    max_len: int = torch.max(lengths).item()  # max_len: scalar
    ids: torch.Tensor = torch.arange(0, max_len, device=lengths.device)  # ids: (max_len,)
    mask: torch.Tensor = ids < lengths.unsqueeze(1)  # lengths.unsqueeze(1): (batch_size, 1), ids < ... : (batch_size, max_len)
    return mask


def to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Moves a tensor to the specified device and ensures it is contiguous.

    This is a utility function often used in data loading and model forwarding
    to place tensors on the correct device (e.g., GPU) and optimize memory access.

    Args:
        x (torch.Tensor): The input tensor to move.
        device (torch.device): The target device (e.g., 'cuda', 'cpu').

    Returns:
        torch.Tensor: The tensor moved to the specified device and made contiguous.

    Examples:
        >>> tensor = torch.randn(2, 3)
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> moved_tensor = to_device(tensor, device)
        >>> print(moved_tensor.device)
        # Output will be 'cuda:0' or 'cpu' depending on availability
    """
    return x.contiguous().to(device, non_blocking=True)


def get_mask_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    """
    Generates a boolean mask from a tensor of lengths. (Duplicate definition, see above)

    Args:
        lengths (torch.Tensor): A 1D tensor of integers representing the lengths
                                of sequences in a batch. Shape: (batch_size,).

    Returns:
        torch.Tensor: A boolean mask tensor. `True` indicates a valid position,
                      `False` indicates a padding position.
                      Shape: (batch_size, max_len).
    """
    max_len: int = torch.max(lengths).item()  # max_len: scalar
    ids: torch.Tensor = torch.arange(0, max_len, device=lengths.device)  # ids: (max_len,)
    mask: torch.Tensor = ids < lengths.unsqueeze(1)  # mask: (batch_size, max_len)
    return mask

def to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Moves a tensor to the specified device and ensures it is contiguous. (Duplicate definition, see above)

    Args:
        x (torch.Tensor): The input tensor to move.
        device (torch.device): The target device.

    Returns:
        torch.Tensor: The tensor moved to the specified device and made contiguous.
    """
    return x.contiguous().to(device, non_blocking=True)


class TextMelCollate:
    """
    Custom collate function for DataLoader that zeros-pads model inputs and targets
    based on the number of frames per step.

    This class is designed to prepare batches of text and mel spectrograms for a
    Tacotron2-VAE model. It handles variable-length sequences by padding them
    to the maximum length within each batch and ensures that mel spectrogram
    lengths are multiples of `n_frames_per_step`.

    Attributes:
        n_frames_per_step (int): The number of mel frames predicted per decoder step.
                                 Used to ensure mel spectrograms are padded to a
                                 multiple of this value.

    Examples:
        >>> from torch.utils.data import DataLoader
        >>> # Assuming a dataset that returns (text_tensor, mel_tensor, speaker_tensor, emotion_tensor)
        >>> class DummyDataset(torch.utils.data.Dataset):
        ...     def __len__(self): return 3
        ...     def __getitem__(self, idx):
        ...         if idx == 0: return torch.randint(0, 50, (10,)), torch.randn(80, 100), torch.tensor(0), torch.tensor(0)
        ...         if idx == 1: return torch.randint(0, 50, (15,)), torch.randn(80, 103), torch.tensor(1), torch.tensor(1)
        ...         if idx == 2: return torch.randint(0, 50, (8,)), torch.randn(80, 95), torch.tensor(0), torch.tensor(2)
        >>> dataset = DummyDataset()
        >>> collate_fn = TextMelCollate(n_frames_per_step=1)
        >>> dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        >>> (text_padded, input_lengths, mel_padded, output_lengths, speaker_padded, emotion_padded) = next(iter(dataloader))
        >>> print(text_padded.shape) # (batch_size, max_text_len)
        >>> print(mel_padded.shape) # (batch_size, n_mel_channels, max_mel_len)
    """
    def __init__(self, n_frames_per_step: int):
        """
        Initializes the TextMelCollate collate function.

        Args:
            n_frames_per_step (int): The number of mel frames generated per decoder step.
                                     Mel spectrograms will be padded to a multiple of this value.
        """
        self.n_frames_per_step: int = n_frames_per_step

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        Processes a batch of data, performing sorting and padding.

        The batch elements are assumed to be tuples of (text, mel, speaker, emotion).
        It sorts the batch by text length in descending order, zero-pads text and mel
        sequences, and ensures mel lengths are multiples of `n_frames_per_step`.

        Args:
            batch (List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
                A list of tuples, where each tuple contains:
                - text (torch.Tensor): 1D tensor of token IDs. Shape: (text_len,).
                - mel (torch.Tensor): 2D tensor of mel spectrograms. Shape: (n_mel_channels, mel_len).
                - speaker (torch.Tensor): 0D tensor (scalar) for speaker ID.
                - emotion (torch.Tensor): 0D tensor (scalar) for emotion ID.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                A tuple containing the padded batch components:
                - text_padded (torch.Tensor): Padded text sequences. Shape: (batch_size, max_text_len).
                - input_lengths (torch.IntTensor): Actual lengths of text sequences. Shape: (batch_size,).
                - mel_padded (torch.Tensor): Padded mel spectrograms. Shape: (batch_size, n_mel_channels, max_mel_len_padded).
                - output_lengths (torch.IntTensor): Actual (padded) lengths of mel spectrograms. Shape: (batch_size,).
                - speaker_padded (torch.Tensor): Stacked speaker IDs. Shape: (batch_size,).
                - emotion_padded (torch.Tensor): Stacked emotion IDs. Shape: (batch_size,).
        """
        # Sorts the batch by text length (descending) for PackedSequence requirement in Tacotron's encoder.
        batch.sort(key=lambda x: x[0].size(0), reverse=True)

        text_padded: List[torch.Tensor] = []
        mel_padded: List[torch.Tensor] = []
        speaker_padded: List[torch.Tensor] = []
        emotion_padded: List[torch.Tensor] = []
        input_lengths: List[int] = []
        output_lengths: List[int] = []
        
        for text, mel, emotion in batch:
            text_padded.append(text)
            input_lengths.append(text.size(0))  # input_lengths: (batch_size,)
            
            # Ensures that the length of the Mel spectrogram is a multiple of n_frames_per_step.
            mel_len: int = mel.size(1)  # mel_len: scalar
            if mel_len % self.n_frames_per_step != 0:
                pad_amt: int = self.n_frames_per_step - (mel_len % self.n_frames_per_step)  # pad_amt: scalar
                mel = torch.nn.functional.pad(mel, (0, pad_amt))  # mel: (n_mel_channels, mel_len + pad_amt)
            
            mel_padded.append(mel)
            output_lengths.append(mel.size(1))  # output_lengths: (batch_size,)
            emotion_padded.append(emotion)

        # Padding of sequences to the maximum length within the batch.
        text_padded = torch.nn.utils.rnn.pad_sequence(text_padded, batch_first=True)  # text_padded: (batch_size, max_text_len)
        # Pad mel sequences and then transpose to (batch_size, n_mel_channels, max_mel_len_padded)
        mel_padded = torch.nn.utils.rnn.pad_sequence(mel_padded, batch_first=True).transpose(1, 2)  # mel_padded: (batch_size, n_mel_channels, max_mel_len_padded)
        
        return (
            text_padded, 
            torch.IntTensor(input_lengths), 
            mel_padded, 
            torch.IntTensor(output_lengths),
            torch.stack(emotion_padded)  # emotion_padded: (batch_size,)
        )
