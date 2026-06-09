import torch


def get_mask_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device)
    mask = ids < lengths.unsqueeze(1)
    return mask


def to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.contiguous().to(device, non_blocking=True)


def get_mask_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device)
    mask = ids < lengths.unsqueeze(1)
    return mask

def to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.contiguous().to(device, non_blocking=True)

class TextMelCollate:
    """ Zeros-pad model inputs and targets based on number of frames per step """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        # Ordena o batch pelo tamanho do texto (decrescente) para o PackedSequence do Tacotron
        batch.sort(key=lambda x: x[0].size(0), reverse=True)

        text_padded, mel_padded, speaker_padded, emotion_padded = [], [], [], []
        input_lengths, output_lengths = [], []
        
        for text, mel, speaker, emotion in batch:
            text_padded.append(text)
            input_lengths.append(text.size(0))
            
            # Garante que o comprimento do Mel seja múltiplo de n_frames_per_step
            mel_len = mel.size(1)
            if mel_len % self.n_frames_per_step != 0:
                pad_amt = self.n_frames_per_step - (mel_len % self.n_frames_per_step)
                mel = torch.nn.functional.pad(mel, (0, pad_amt))
            
            mel_padded.append(mel)
            output_lengths.append(mel.size(1))
            speaker_padded.append(speaker)
            emotion_padded.append(emotion)

        # Padding das sequências
        text_padded = torch.nn.utils.rnn.pad_sequence(text_padded, batch_first=True)
        mel_padded = torch.nn.utils.rnn.pad_sequence(mel_padded, batch_first=True).transpose(1, 2)
        
        return (
            text_padded, 
            torch.IntTensor(input_lengths), 
            mel_padded, 
            torch.IntTensor(output_lengths),
            torch.stack(speaker_padded), 
            torch.stack(emotion_padded)
        )