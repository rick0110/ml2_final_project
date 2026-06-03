import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def log_gst_attention_heatmap(tb_logger, att_weights, epoch, prefix="interpretability/"):
    """
    Gera um heatmap mostrando quais Style Tokens estão sendo ativados
    para as primeiras amostras do batch de validação.
    """
    # att_weights tem shape: [batch_size, num_tokens]
    # Vamos pegar até 8 amostras do batch para visualizar
    num_samples = min(8, att_weights.size(0))
    weights = att_weights[:num_samples].detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(weights, ax=ax, annot=True, cmap="viridis", fmt=".2f", vmin=0.0, vmax=1.0)
    
    ax.set_title(f"GST Attention Weights (Batch Samples) - Epoch {epoch}")
    ax.set_xlabel("Style Token Index")
    ax.set_ylabel("Batch Sample Index")
    
    # Acessa o SummaryWriter nativo dentro do seu TensorBoardLogger
    # Ajuste 'tb_logger.writer' se o seu wrapper usar outro nome de atributo
    tb_logger.writer.add_figure(f"{prefix}attention_heatmap", fig, global_step=epoch)
    plt.close(fig)

def log_style_token_similarity(tb_logger, gst_module, epoch, prefix="interpretability/"):
    """
    Calcula a similaridade de cosseno entre todos os style tokens para 
    garantir que eles estão aprendendo características distintas (evitar mode collapse).
    """
    tokens = gst_module.style_tokens.detach() # shape: [num_tokens, hidden_size]
    
    # Normaliza os tokens e calcula a matriz de similaridade
    tokens_norm = F.normalize(tokens, p=2, dim=1)
    similarity_matrix = torch.matmul(tokens_norm, tokens_norm.T).cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(similarity_matrix, ax=ax, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1.0, vmax=1.0)
    
    ax.set_title(f"Style Tokens Cosine Similarity - Epoch {epoch}")
    ax.set_xlabel("Token Index")
    ax.set_ylabel("Token Index")
    
    tb_logger.writer.add_figure(f"{prefix}token_similarity", fig, global_step=epoch)
    plt.close(fig)

def log_style_token_embeddings(tb_logger, gst_module, epoch):
    """
    Plota os embeddings dos tokens no Projector interativo do TensorBoard.
    """
    tokens = gst_module.style_tokens.detach()
    # Adiciona metadados para que apareça como "Token 0", "Token 1", etc. no Projector
    metadata = [f"Token_{i}" for i in range(tokens.size(0))]
    tb_logger.writer.add_embedding(tokens, metadata=metadata, global_step=epoch, tag="GST_Tokens_Latent_Space")