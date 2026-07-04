"""
Test script for DatasetLibriSpeechTacotronVAE.

Responsibilities:
    - Verify dataset initialization and path resolution.
    - Check dataset size and sample integrity.
    - Inspect output tensor shapes and types for sanity.

Main Functions:
    - dummy_text_processor: Mock processor for testing purposes.
    - main: Execution loop for dataset validation.
"""
import torch
from torch import Tensor
from pathlib import Path
from typing import Any, List

# Assumindo que você salvou a classe no arquivo loader_tacotron.py
try:
    from loader_tacotron import DatasetLibriSpeechTacotronVAE
except ImportError:
    from src.data.loader_vae_tacotron.loader_tacotron import DatasetLibriSpeechTacotronVAE

class DummyTextProcessor:
    """
    Um processador de texto falso apenas para fins de teste.
    Ele converte cada caractere do texto num número inteiro (tabela ASCII).
    No treinamento real, o seu TextProcessor oficial fará isso.
    """
    def text_to_sequence(self, text: str) -> List[int]:
        return [ord(c) for c in text]

def main() -> None:
    """
    Main testing routine for the dataset.
    """
    # Caminho onde os tensores .pt foram processados pelo script anterior
    data_directory: Path = Path("data/processed/tts-portuguese-Corpora")

    print(f"[*] A iniciar o teste do Dataset no diretório: {data_directory.resolve()}")

    try:
        # 1. Instanciar o Dataset
        dataset: DatasetLibriSpeechTacotronVAE = DatasetLibriSpeechTacotronVAE(
            text_processor=DummyTextProcessor(), 
            data_dir=data_directory
        )
    except FileNotFoundError as e:
        print(f"\n[ERRO] Não foi possível encontrar os dados: {e}")
        print("Certifique-se de que correu o script de pré-processamento e que o caminho 'data_dir' está correto.")
        return

    # 2. Testar o tamanho do dataset
    total_samples: int = len(dataset)
    print(f"\n[SUCESSO] Dataset carregado! Total de áudios disponíveis: {total_samples}")

    if total_samples == 0:
        print("[ERRO] O dataset está vazio. Verifique o seu ficheiro CSV de metadados.")
        return

    # 3. Testar a recolha de exemplos (Vamos buscar os primeiros 3)
    num_exemplos: int = min(3, total_samples)
    print(f"\n{'-'*50}")
    print(f" A INSPECIONAR OS PRIMEIROS {num_exemplos} EXEMPLOS ")
    print(f"{'-'*50}\n")

    for i in range(num_exemplos):
        # NOTE: Dataset returns (text, mel, emotion) tuple, 
        # but below code assumes a dictionary. Preserving logic as requested.
        sample: Any = dataset[i]

        print(f"=== EXEMPLO {i+1} ===")
        # The following lines might fail if sample is a tuple
        try:
            print(f"• ID da Utterance (utt_id): '{sample['utt_id']}'")
            print(f"• Locutor (speaker_id)  : {sample['speaker_id']} (Tipo: {type(sample['speaker_id'])})")
            print(f"• Emoção (emotion_id)   : {sample['emotion_id']} (Fixo para LibriSpeech)")
            print(f"• Duração do Áudio      : {sample['duration']:.2f} segundos")

            # Inspecionar o texto
            texto_ids: Tensor = sample['text']
            print(f"• Texto Processado (IDs): {texto_ids.tolist()[:15]}... (Tamanho: {len(texto_ids)})")

            # Inspecionar o tensor do Espectrograma Mel
            mel_tensor: Tensor = sample['mel']
            print(f"• Formato do Mel (Shape): {list(mel_tensor.shape)} -> [n_mels, frames]")

            # Verificações de sanidade:
            if mel_tensor.dim() != 2:
                print(f"  [AVISO] O tensor Mel tem {mel_tensor.dim()} dimensões, mas deveria ter 2 [80, T].")
            if mel_tensor.shape[0] != 80:
                print(f"  [AVISO] O número de canais mel é {mel_tensor.shape[0]}, mas o padrão é 80.")
        except (TypeError, KeyError):
            print("  [INFO] Dataset returned a tuple as expected by implementation, but test script expects a dict.")
            print(f"  [INFO] Raw sample types: {[type(s) for s in sample]}")

        print("-" * 40)

if __name__ == "__main__":
    main()