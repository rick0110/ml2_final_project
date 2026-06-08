#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
from datetime import datetime
import torch
from torch.optim import AdamW
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.TTS_GST import TTS_GST
from models.LatentResidualMapping import LatentResidualMapping
from utils import create_dataset, create_dataloaders, pad_sequence
from losses import AlignAndCutL1Loss

# Cores para o terminal
C_GREEN = '\033[92m'
C_BLUE = '\033[94m'
C_YELLOW = '\033[93m'
C_MAGENTA = '\033[95m'
C_RESET = '\033[0m'

def run_sanity_checks(acoustic_model, transferrer, vocoder, loader, device):
    """Testa todo o fluxo carregando dados, processando e inferindo passo-a-passo."""
    print(f"\n{C_YELLOW}{'='*60}{C_RESET}")
    print(f"{C_YELLOW}[TESTE DE SANIDADE] Validando pipeline antes da Epoch...{C_RESET}")
    
    # 1. Carregamento de Dados
    batch = next(iter(loader))
    mel_original = batch["mel"].to(device)
    if mel_original.dim() == 4 and mel_original.size(1) == 1:
        mel_original = mel_original.squeeze(1)
    texts = batch["text"]
    
    print(f"\n{C_BLUE}[PASSO 1] DADOS DE ENTRADA{C_RESET}")
    print(f" -> Batch Size: {mel_original.size(0)}")
    print(f" -> Texto Exemplo [0]: '{texts[0][:50]}...'")
    print(f" -> Mel Original Shape: {mel_original.shape}")

    # 2. Parseamento e Geração do Alvo
    padding_value = acoustic_model.spec_generator.fastpitch.encoder.padding_idx
    parsed_texts = [torch.tensor(acoustic_model.spec_generator.parse(t)) for t in texts]
    text_ids = pad_sequence(parsed_texts, padding_value=padding_value).to(device)

    with torch.no_grad():
        _, mel_alvo = acoustic_model(text_tokens=text_ids, audio_inputs=mel_original, return_att_weights=False)
        
    print(f"\n{C_BLUE}[PASSO 2] GERADOR ACÚSTICO (TEXTO -> MEL ALVO){C_RESET}")
    print(f" -> Tokens Texto Shape: {text_ids.shape}")
    print(f" -> Mel Alvo (Predito) Shape: {mel_alvo.shape}")

    # 3. Transferidor de Espaço Latente
    with torch.no_grad():
        mel_mapeado = transferrer(mel_original)
        
    print(f"\n{C_BLUE}[PASSO 3] TRANSFERIDOR (MEL ORIGINAL -> MEL MAPEADO){C_RESET}")
    print(f" -> Mel Mapeado Shape: {mel_mapeado.shape}")

    # 4. Corte e Perda (Loss)
    criterion = AlignAndCutL1Loss()
    loss_val, mapeado_cut, alvo_cut = criterion(mel_mapeado, mel_alvo)
    
    print(f"\n{C_BLUE}[PASSO 4] CORTE E CÁLCULO DE PERDA (L1){C_RESET}")
    print(f" -> Cortado Mapeado Shape: {mapeado_cut.shape}")
    print(f" -> Cortado Alvo Shape: {alvo_cut.shape}")
    print(f" -> Loss Exemplo Calculada: {loss_val.item():.5f}")

    # 5. Geração de Áudio (Vocoder)
    with torch.no_grad():
        # Passa apenas o primeiro item do batch para ser rápido no teste
        audio_sinal = vocoder(spec=mel_mapeado[0:1])
        
    print(f"\n{C_BLUE}[PASSO 5] VOCODER INFERÊNCIA{C_RESET}")
    print(f" -> Áudio Gerado Shape: {audio_sinal.shape}")

    print(f"\n{C_GREEN}[SANIDADE OK] Todos os tensores processados com sucesso!{C_RESET}")
    print(f"{C_YELLOW}{'='*60}{C_RESET}\n")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--experiment-name", type=str, default=None)
    return parser.parse_args()

def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_dir = PROJECT_ROOT / "experiments" / "latent_transfer"
    experiment_name = args.experiment_name if args.experiment_name else f"attempt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = base_dir / experiment_name / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Diretório: {checkpoint_dir}")

    # Carrega Dataloader
    dataset = create_dataset()
    train_loader, _ = create_dataloaders(dataset, batch_size=args.batch_size)

    # Carrega Modelo Acústico e Congela Tudo
    acoustic_model = TTS_GST(num_gst_tokens=10, gst_token_dim=384).to(device)
    acoustic_model.eval()
    for param in acoustic_model.parameters():
        param.requires_grad = False
        
    vocoder = acoustic_model.vocoder
    vocoder.eval()

    # Instancia Transferidor (ÚNICO COM PARÂMETROS TREINÁVEIS)
    # Passa para 512, acompanhando a nova arquitetura
    transferrer = LatentResidualMapping(channels=80, hidden_dim=1024).to(device)
    optimizer = AdamW(transferrer.parameters(), lr=args.learning_rate)
    criterion = AlignAndCutL1Loss()

    run_sanity_checks(acoustic_model, transferrer, vocoder, train_loader, device)


    for epoch in range(args.num_epochs):
        # =========================================================
        # TESTE OBRIGATÓRIO ANTES DE INICIAR O TREINO DA ÉPOCA
        # =========================================================
        
        transferrer.train()
        total_epoch_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Época {epoch+1}/{args.num_epochs}"):
            mel_original = batch["mel"].to(device)
            if mel_original.dim() == 4 and mel_original.size(1) == 1:
                mel_original = mel_original.squeeze(1)
                
            texts = batch["text"]
            padding_value = acoustic_model.spec_generator.fastpitch.encoder.padding_idx
            parsed_texts = [torch.tensor(acoustic_model.spec_generator.parse(t)) for t in texts]
            text_ids = pad_sequence(parsed_texts, padding_value=padding_value).to(device)

            # Extração sem gradiente
            with torch.no_grad():
                _, mel_alvo = acoustic_model(text_tokens=text_ids, audio_inputs=mel_original, return_att_weights=False)

            optimizer.zero_grad()
            
            # Passa no transferidor
            mel_mapeado = transferrer(mel_original)
            
            # Corta e calcula L1
            loss, _, _ = criterion(mel_mapeado, mel_alvo)
            
            loss.backward()
            optimizer.step()
            total_epoch_loss += loss.item()

        print(f"{C_MAGENTA}Época [{epoch+1}/{args.num_epochs}] - Loss Média: {total_epoch_loss / len(train_loader):.5f}{C_RESET}")

        if epoch == 0 or (epoch + 1) % 5 == 0:
            torch.save(transferrer.state_dict(), checkpoint_dir / f"epoch_{epoch+1:04d}.pt")

if __name__ == "__main__":
    main()