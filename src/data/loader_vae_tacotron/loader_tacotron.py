import csv
import re
import sys
import torch
from pathlib import Path
from torch.utils.data import Dataset, random_split



from num2words import num2words

ROOT_DIR = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "src" / "models" / "tacotron2_vae"))

from layers import TacotronSTFT

MAX_WAV_VALUE = 32768.0



class TextNormalizerEN:
    def __init__(self):
        self.number_pattern = re.compile(r'\d+')
        
    def _replace_numbers(self, match):
        if num2words is None:
            return match.group(0)
        number = int(match.group(0))
        return num2words(number, lang='en')

    def normalize(self, text: str) -> str:
        text = self.number_pattern.sub(self._replace_numbers, text)
        replaces = {'–': '-', '—': '-', '−': '-', '·': '', '"': '', '\'': ''}
        for old_char, new_char in replaces.items():
            text = text.replace(old_char, new_char)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()


class DatasetLibriSpeechTacotronVAE(Dataset):
    def __init__(self, text_processor, data_dir=Path("data/processed/libriSpeech-en-tacotron-vae")):
        self.data_dir = Path(data_dir)
        self.metadata_path = self.data_dir / "mels_metadata.csv"
        self.normalizer = TextNormalizerEN()
        self.text_processor = text_processor 
        self.files = self._load_files_list()
        self.stft = TacotronSTFT(filter_length=800,
                                 hop_length=200,
                                 win_length=800, 
                                 sampling_rate=22050, 
                                 mel_fmin=0.0, mel_fmax=8000.0)


    def _load_files_list(self):
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def __len__(self):
        return len(self.files)
    
    def get_mel(self, audio):
        audio_norm = audio / MAX_WAV_VALUE
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec



    def __getitem__(self, idx):
        row = self.files[idx]
        sample = torch.load(row["mel_path"], map_location="cpu", weights_only=False)
        
        clean_text = self.normalizer.normalize(row["text"])
        sequence_list = self.text_processor.text_to_sequence(clean_text)
        
        text_sequence = torch.LongTensor(sequence_list)
        
        audio = sample["waveform"].squeeze(0)
        # ensure match dimentions [80, T]
        mel_tensor = self.get_mel(audio)
        
        emotion = torch.zeros(4, dtype=torch.float32)
        emotion[0] = 1.0  # on neutral emotions: it will be used when fine-tuning on VERBO.

        # Return text sequence, mel spectrogram, and emotion label
        return text_sequence, mel_tensor, emotion
    
def load_data(
    text_processor,
    data_dir=Path("data/processed/libriSpeech-en-tacotron-vae"),
    val_split=0.1,
    generator=None,
):
    Dataset = DatasetLibriSpeechTacotronVAE(text_processor=text_processor, data_dir=data_dir)
    data_train, data_test, data_val = random_split(Dataset, [len(Dataset) - 2*int(len(Dataset) * val_split // 2), int(len(Dataset) * val_split // 2), int(len(Dataset) * val_split // 2)], generator=generator)

    return data_train, data_test, data_val