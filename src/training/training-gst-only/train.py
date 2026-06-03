#!/usr/bin/env python3
"""First-step TTS model training script with GST Interpretability.

Usage:
    python train.py --num-epochs 100 --batch-size 32 --learning-rate 1e-3
"""

import sys
import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.train_first_step.text_processing import BatchTextTokenizer
from train_utils import train_epoch, validate_epoch, save_checkpoint, load_checkpoint, TensorBoardLogger, log_validation_audio_examples
from losses import CombinedTTSLoss
from models.TTS_GST import TTS_GST
from utils import create_experiment_dir, create_dataset, create_dataloaders

# Importando os novos utilitários de interpretabilidade
from training.train_first_step.interpretability_utils import (
    log_gst_attention_heatmap, 
    log_style_token_similarity, 
    log_style_token_embeddings
)

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--scheduler-patience", type=int, default=3)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--weight-reconstruction", type=float, default=1.0)
    parser.add_argument("--weight-diversity", type=float, default=0.5)
    parser.add_argument("--diversity-margin", type=float, default=0.1)
    parser.add_argument("--style-embedding-dim", type=int, default=384)
    parser.add_argument("--num-style-tokens", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume-experiment", type=str, default=None)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_arguments()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    experiment_dir = Path(args.resume_experiment) if args.resume_experiment else create_experiment_dir(args.experiment_name)
    checkpoint_dir = experiment_dir / "checkpoints"
    tensorboard_dir = experiment_dir / "tensorboard"
    
    dataset = create_dataset()
    train_loader, val_loader = create_dataloaders(dataset, args.batch_size, args.num_workers, args.val_split, args.seed)
    tokenizer = BatchTextTokenizer()
    
    model = TTS_GST(
        num_gst_tokens=args.num_style_tokens,
        gst_token_dim=args.style_embedding_dim,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=args.scheduler_factor, patience=args.scheduler_patience, min_lr=args.scheduler_min_lr)
    criterion = CombinedTTSLoss(weight_reconstruction=args.weight_reconstruction, weight_diversity=args.weight_diversity, diversity_margin=args.diversity_margin).to(device)
    
    tb_logger = TensorBoardLogger(tensorboard_dir)
    tb_logger.log_model_info(model)
    
    start_epoch = 0
    best_val_loss = float("inf")

    # Tratamento para retomar treinamento
    if args.resume:
        checkpoint_path = checkpoint_dir / args.resume
        if checkpoint_path.exists():
            start_epoch, best_val_loss = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
            print(f"Resumed from checkpoint {checkpoint_path} at epoch {start_epoch} with best val loss {best_val_loss:.4f}")
        else:
            print(f"Checkpoint {checkpoint_path} not found. Starting from scratch.")
    
    elif args.resume_experiment:
        best_checkpoint = checkpoint_dir / "best.pt" # Correção do erro de sintaxe aqui
        if best_checkpoint.exists():
            start_epoch, best_val_loss = load_checkpoint(model, optimizer, scheduler, best_checkpoint)
            print(f"Resumed from best checkpoint {best_checkpoint} at epoch {start_epoch} with val loss {best_val_loss:.4f}")
        else:
            print(f"Best checkpoint {best_checkpoint} not found. Starting from scratch.")


    for epoch in range(start_epoch, args.num_epochs):
        # ATENÇÃO: As funções train_epoch e validate_epoch precisarão ser 
        # atualizadas no seu arquivo `train_utils.py` para esperar uma 
        # tupla (audio_outputs, att_weights) do modelo.
        train_metrics = train_epoch(model, tokenizer, train_loader, optimizer, criterion, device, epoch, args.num_epochs)
        val_metrics = validate_epoch(model, tokenizer, val_loader, criterion, device, epoch, args.num_epochs)
        scheduler.step(val_metrics["loss"])

        # Logs visuais, de áudio e interpretabilidade (ex: a cada época)
        if epoch == 0 or (epoch + 1) % 1 == 0:
            example_batch = next(iter(val_loader))
            # Você precisará passar apenas 'model' caso o vocoder já esteja embutido (como parece ser no TTS_GST)
            log_validation_audio_examples(model, model.vocoder, example_batch, device, tb_logger, epoch)
            
            # --- SEÇÃO DE INTERPRETABILIDADE GST ---
            model.eval()
            with torch.no_grad():
                # Assumindo que seu batch gera 'text_embed' e 'audio_inputs'
                # Você precisará adaptar isso conforme os nomes devolvidos pelo seu DataLoader
                text_input, audio_input = example_batch[0].to(device), example_batch[1].to(device)
                
                # Fazendo forward de validação para capturar pesos de atenção específicos
                # Modifique caso seu TTS_GST precise de embeddings de texto já convertidos
                _, att_weights = model(text_input, audio_input)
                
                # 1. Heatmap da Atenção (para ver se áudios diferentes ativam tokens diferentes)
                log_gst_attention_heatmap(tb_logger, att_weights, epoch)
                
                # 2. Similaridade entre Tokens (para monitorar Mode Collapse)
                log_style_token_similarity(tb_logger, model.gst, epoch)
                
                # 3. Log dos embeddings no Projector 3D do TensorBoard
                log_style_token_embeddings(tb_logger, model.gst, epoch)
            # ----------------------------------------

        tb_logger.log_metrics(train_metrics, epoch, prefix="train/")
        tb_logger.log_metrics(val_metrics, epoch, prefix="val/")
        tb_logger.flush()
        
        save_checkpoint(model, optimizer, scheduler, epoch + 1, train_metrics, checkpoint_dir, f"epoch_{epoch+1:04d}.pt")
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(model, optimizer, scheduler, epoch + 1, val_metrics, checkpoint_dir, "best.pt")
            
    tb_logger.close()

if __name__ == "__main__":
    main()