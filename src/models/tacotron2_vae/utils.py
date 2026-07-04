"""
Utility functions for Tacotron 2 VAE model.

Responsibilities:
    - Generate attention masks from sequence lengths.
    - Move tensors to specified devices with contiguous memory.
    - Implement TextMelCollate for batching text and mel spectrograms.

Main Classes:
    - TextMelCollate: Collate function for DataLoader that handles variable-length padding.

Main Functions:
    - get_mask_from_lengths: Create boolean mask for padded sequences.
    - to_device: Utility to move tensors to GPU/CPU.

Tensor Conventions:
    B = batch size
    T = sequence length (frames/tokens)
    n_mels = mel frequency bins
"""
import torch
from torch import Tensor
from typing import List, Tuple, Any


def get_mask_from_lengths(lengths: Tensor) -> Tensor:
    """
    Generates a boolean mask from a tensor of lengths.

    Args:
        lengths (Tensor): Sequence lengths.
            Shape: (B,)

    Returns:
        Tensor: Boolean mask (True for valid, False for padding).
            Shape: (B, T_max)

    Example:
        >>> lengths = torch.tensor([3, 5, 2])
        >>> mask = get_mask_from_lengths(lengths)
    """
    max_len: int = torch.max(lengths).item()
    ids: Tensor = torch.arange(0, max_len, device=lengths.device) # (T_max,)
    mask: Tensor = ids < lengths.unsqueeze(1) # (B, T_max)
    return mask


def to_device(x: Tensor, device: torch.device) -> Tensor:
    """
    Moves a tensor to the specified device and ensures it is contiguous.

    Args:
        x (Tensor): Input tensor.
        device (torch.device): Target device.

    Returns:
        Tensor: Tensor on the target device.
    """
    return x.contiguous().to(device, non_blocking=True)


class TextMelCollate:
    """
    Custom collate function for DataLoader that zeros-pads model inputs and targets.

    Architecture:
        Sorts batch by text length -> Pads text -> Pads mel (ensures multiple of frames_per_step).

    Inputs:
        batch:
            List of (text, mel, emotion) tuples.
            text: (T_text,)
            mel: (n_mels, T_mel)
            emotion: scalar

    Outputs:
        text_padded: (B, max_T_text)
        input_lengths: (B,)
        mel_padded: (B, n_mels, max_T_mel_padded)
        output_lengths: (B,)
        emotion_padded: (B,)

    Example:
        >>> collate_fn = TextMelCollate(n_frames_per_step=1)
        >>> dataloader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)
    """
    def __init__(self, n_frames_per_step: int) -> None:
        """
        Initializes the TextMelCollate.

        Args:
            n_frames_per_step (int): Multiple to which mel frames should be padded.
        """
        self.n_frames_per_step: int = n_frames_per_step

    def __call__(self, batch: List[Tuple[Tensor, Tensor, Tensor]]) -> Tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor
    ]:
        """
        Processes a batch of data, performing sorting and padding.

        Args:
            batch (List[Tuple[Tensor, Tensor, Tensor]]): List of (text, mel, emotion).

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: Padded and stacked batch components.
        """
        # Sorts the batch by text length (descending)
        batch.sort(key=lambda x: x[0].size(0), reverse=True)

        text_padded_list: List[Tensor] = []
        mel_padded_list: List[Tensor] = []
        emotion_padded_list: List[Tensor] = []
        input_lengths_list: List[int] = []
        output_lengths_list: List[int] = []
        
        for text, mel, emotion in batch:
            text_padded_list.append(text)
            input_lengths_list.append(text.size(0))
            
            # Ensures that the length of the Mel spectrogram is a multiple of n_frames_per_step.
            mel_len: int = mel.size(1)
            if mel_len % self.n_frames_per_step != 0:
                pad_amt: int = self.n_frames_per_step - (mel_len % self.n_frames_per_step)
                mel = torch.nn.functional.pad(mel, (0, pad_amt)) # (n_mels, T_mel + pad)
            
            mel_padded_list.append(mel)
            output_lengths_list.append(mel.size(1))
            emotion_padded_list.append(emotion)

        # Padding
        text_padded: Tensor = torch.nn.utils.rnn.pad_sequence(text_padded_list, batch_first=True) # (B, max_T_text)
        
        # Transpose each tensor to (T_mel, n_mels)
        mel_transposed_list = [mel.transpose(0, 1) for mel in mel_padded_list]
        
        # Pad sequence over T_mel dimension: shape (B, max_T_mel_padded, n_mels)
        mel_padded_temp = torch.nn.utils.rnn.pad_sequence(mel_transposed_list, batch_first=True)
        
        # Transpose back to (B, n_mels, max_T_mel_padded)
        mel_padded: Tensor = mel_padded_temp.transpose(1, 2)
        
        input_lengths: Tensor = torch.IntTensor(input_lengths_list) # (B,)
        output_lengths: Tensor = torch.IntTensor(output_lengths_list) # (B,)
        emotion_padded: Tensor = torch.stack(emotion_padded_list) # (B,)

        return (
            text_padded, 
            input_lengths, 
            mel_padded, 
            output_lengths,
            emotion_padded
        )
