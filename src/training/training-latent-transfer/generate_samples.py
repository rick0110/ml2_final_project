#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
import torch
import numpy as np
import scipy.io.wavfile as wavf

# Mapeia as raízes do projeto
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# =====================================================================
# IMPORTAÇÃO DOS MODELOS
# =====================================================================
from models.HiFi_GAN import load_hifigan_model
from models.LatentResidualMapping import LatentResidualMapping
from utils import create_dataset, create_dataloaders

# Cores para o terminal
C_GREEN = '\033[92m'
C_BLUE = '\033[94m'
C_YELLOW = '\033[93m'
C_RESET = '\033[0m'

def save_audio(tensor, filepath, sample_rate=22050):
    """Normaliza e guarda o tensor de áudio do HiFi-GAN num ficheiro .wav"""
    if isinstance(tensor, tuple): 
        tensor = tensor[0]  
    
    audio = tensor.squeeze().cpu().numpy()
    # Normalização de segurança contra picos (clipping)
    audio = audio / (np.max(np.abs(audio)) + 1e-6)
    # Conversão para formato standard de áudio int16
    audio_int16 = (audio * 32767).astype(np.int16)
    wavf.write(filepath, sample_rate, audio_int16)

def parse_args():
    parser = argparse.ArgumentParser(description="Geração de Amostras para Comparação (FastPitch vs Transferidor)")
    parser.add_argument("--transfer-ckpt", type=str, default="experiments/latent_transfer/transfer_latent/checkpoints/epoch_0050.pt", help="Checkpoint do Transferidor treinado (.pt)")
    parser.add_argument("--num-samples", type=int, default=5, help="Quantidade de amostras a gerar do dataset")
    parser.add_argument("--out-dir", type=str, default="exports/samples", help="Pasta para guardar os .wav")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepara a pasta de saída
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{C_YELLOW}=== INICIANDO GERAÇÃO DE AMOSTRAS DE ÁUDIO ==={C_RESET}")
    print(f"A guardar as amostras em: {out_dir}\n")

    # 1. Carregar Modelos via NeMo / load_hifigan_model
    print(f"{C_BLUE}A carregar FastPitch e HiFi-GAN do local_weight_models...{C_RESET}")
    fastpitch, vocoder = load_hifigan_model(freeze=True)
    fastpitch = fastpitch.to(device)
    vocoder = vocoder.to(device)
    
    print(f"{C_BLUE}A carregar Transferidor Latente...{C_RESET}")
    transferrer = LatentResidualMapping(channels=80, hidden_dim=256).to(device)
    if Path(args.transfer_ckpt).exists():
        transferrer.load_state_dict(torch.load(args.transfer_ckpt, map_location=device))
        print(" -> Pesos do transferidor carregados com sucesso!")
    else:
        print(f" -> {C_YELLOW}AVISO: Checkpoint do transferidor não encontrado. A usar pesos não treinados.{C_RESET}")
    transferrer.eval()

    # 2. Carregar Dados
    print(f"{C_BLUE}A extrair {args.num_samples} frases do Dataset...{C_RESET}\n")
    dataset = create_dataset()
    
    # CORREÇÃO: Desempacotar a tupla (train_loader, val_loader)
    loader, _ = create_dataloaders(dataset, batch_size=args.num_samples)
    batch = next(iter(loader))
    
    mels_originais = batch["mel"].to(device)
    if mels_originais.dim() == 4 and mels_originais.size(1) == 1:
        mels_originais = mels_originais.squeeze(1)
    textos = batch["text"]

    # 3. Geração e Inferência
    with torch.no_grad():
        for i in range(args.num_samples):
            texto_atual = textos[i]
            print(f"{C_YELLOW}Processando Amostra {i+1}: {C_RESET}{texto_atual[:60]}...")
            
            # Corta o mel correspondente isolando-o do batch
            mel_orig = mels_originais[i:i+1]
            
            # =========================================================
            # CAMINHO 1: TEXTO -> FASTPITCH -> VOCODER
            # =========================================================
            parsed_text = fastpitch.parse(texto_atual)
            if isinstance(parsed_text, tuple): parsed_text = parsed_text[0]
            
            # Adiciona dimensão de batch
            if parsed_text.dim() == 1:
                parsed_text = parsed_text.unsqueeze(0)
            parsed_text = parsed_text.to(device)
            
            # Gera Mel via FastPitch
            if hasattr(fastpitch, 'generate_spectrogram'):
                mel_fastpitch = fastpitch.generate_spectrogram(tokens=parsed_text)
            else:
                mel_fastpitch = fastpitch(text_tokens=parsed_text)[0]
                
            if isinstance(mel_fastpitch, tuple): mel_fastpitch = mel_fastpitch[0]
            
            # Converte Mel FastPitch para Áudio via Vocoder
            if hasattr(vocoder, 'convert_spectrogram_to_audio'):
                audio_caminho_1 = vocoder.convert_spectrogram_to_audio(spec=mel_fastpitch)
            else:
                audio_caminho_1 = vocoder(spec=mel_fastpitch)

            # =========================================================
            # CAMINHO 2: MEL ORIGINAL -> TRANSFERIDOR -> VOCODER
            # =========================================================
            mel_transferido = transferrer(mel_orig)
            
            if hasattr(vocoder, 'convert_spectrogram_to_audio'):
                audio_caminho_2 = vocoder.convert_spectrogram_to_audio(spec=mel_transferido)
            else:
                audio_caminho_2 = vocoder(spec=mel_transferido)

            # =========================================================
            # GUARDAR OS FICHEIROS NO DISCO
            # =========================================================
            prefixo = out_dir / f"amostra_{i+1:02d}"
            save_audio(audio_caminho_1, f"{prefixo}_1_FastPitch_Sem_Transferidor.wav")
            save_audio(audio_caminho_2, f"{prefixo}_2_Com_Transferidor.wav")

    print(f"\n{C_GREEN}=== SUCESSO! ==={C_RESET}")
    print(f"Os teus ficheiros .wav foram gerados na pasta: {out_dir}")

if __name__ == "__main__":
    main()