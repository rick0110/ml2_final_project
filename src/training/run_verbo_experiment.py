import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 1. Ajuste de Rota Mestra
caminho_script = os.path.dirname(os.path.abspath(__file__))
raiz_projeto = os.path.abspath(os.path.join(caminho_script, "../.."))

if raiz_projeto not in sys.path:
    sys.path.append(raiz_projeto)

from src.training.verbo_dataset import VerboDataset

# --- A NOVIDADE: FUNÇÃO DE PREENCHIMENTO (PADDING) ---
def verbo_collate_fn(batch):
    """
    Pega uma lista de itens do dataset e os empilha.
    Se os espectrogramas tiverem tamanhos de tempo diferentes, preenche com silêncio.
    """
    # 1. Encontra o maior tamanho de tempo (dimensão 1) no lote atual
    max_time_len = max([item["mel"].shape[1] for item in batch])
    
    padded_mels = []
    emotion_ids = []
    speakers = []
    texts = []
    
    for item in batch:
        mel = item["mel"] # Shape: [80, tempo_variavel]
        
        # 2. Calcula quanto falta de "silêncio" para chegar no maior tamanho
        pad_amount = max_time_len - mel.shape[1]
        
        # 3. Preenche (pad) à direita no eixo do tempo. 
        # O valor -11.51 é aprox. o logaritmo de zero (log(1e-5)), representando silêncio absoluto.
        padded_mel = F.pad(mel, (0, pad_amount), value=-11.51)
        
        padded_mels.append(padded_mel)
        emotion_ids.append(item["emotion_id"])
        speakers.append(item["speaker"])
        texts.append(item["text"])
        
    return {
        # Empilha (stack) todos os tensores que agora têm exatamente o mesmo tamanho
        "mel": torch.stack(padded_mels),
        "emotion_id": torch.tensor(emotion_ids, dtype=torch.long),
        "speaker": speakers,
        "text": torch.stack(texts)
    }
# -----------------------------------------------------

def main():
    dataset_root = os.path.join(raiz_projeto, "src", "data")
    print(f"Iniciando mapeamento do VERBO-Dataset a partir de: {dataset_root}/Audios")
    
    try:
        verbo_data = VerboDataset(dataset_root=dataset_root, target_sample_rate=22050)
        print(f"Sucesso! Encontrados {len(verbo_data)} arquivos de áudio válidos.")
        
        # INJETAMOS A FUNÇÃO DE PADDING AQUI NO DataLoader
        verbo_loader = DataLoader(
            verbo_data, 
            batch_size=4, 
            shuffle=True, 
            collate_fn=verbo_collate_fn  # O PyTorch agora usará nossa regra para empilhar!
        )
        
        for batch in verbo_loader:
            print("\n--- Inspeção do Primeiro Lote (Batch) ---")
            print(f"Shape do Espectrograma Mel: {batch['mel'].shape}") 
            print(f"IDs numéricos das Emoções: {batch['emotion_id']}")
            print(f"Locutores: {batch['speaker']}")
            print("-----------------------------------------\n")

            print("Gerando imagem do espectrograma colorido...")
            
            from PIL import Image
            import numpy as np

            # Extrai o primeiro espectrograma do lote
            mel_tensor = batch['mel'][0]
            locutor = batch['speaker'][0]
            emocao = batch['emotion_id'][0].item()

            # Normaliza a matriz matemática para valores decimais entre 0.0 e 1.0
            mel_min = mel_tensor.min()
            mel_max = mel_tensor.max()
            mel_norm = (mel_tensor - mel_min) / (mel_max - mel_min)
            mel_norm_np = mel_norm.cpu().numpy()

            # Inverte o eixo Y para frequências graves ficarem embaixo
            mel_norm_np = np.flipud(mel_norm_np)

            # --- CRIANDO O MAPA DE CORES ESTILO "JET" MANUALMENTE ---
            # Âncoras de intensidade: 0.0 = silêncio absoluto, 1.0 = pico de voz
            x = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
            
            # Valores de Vermelho, Verde e Azul para cada âncora
            r_pts = np.array([0,   0,   100, 255, 255])
            g_pts = np.array([0,   180, 255, 200, 0  ])
            b_pts = np.array([100, 255, 50,  0,   0  ])

            # Interpola os valores da matriz para encontrar as cores exatas
            r = np.interp(mel_norm_np, x, r_pts).astype(np.uint8)
            g = np.interp(mel_norm_np, x, g_pts).astype(np.uint8)
            b = np.interp(mel_norm_np, x, b_pts).astype(np.uint8)

            # Empilha os 3 canais (RGB)
            rgb_matrix = np.stack([r, g, b], axis=-1)

            # Cria a imagem colorida
            img = Image.fromarray(rgb_matrix, mode='RGB')
            
            # Estica a imagem para ficar com a proporção retangular clássica
            largura_nova = rgb_matrix.shape[1] * 2
            altura_nova = rgb_matrix.shape[0] * 4
            img = img.resize((largura_nova, altura_nova), Image.NEAREST)

            # Salva o arquivo na raiz
            nome_arquivo = f'espectrograma_colorido_{locutor}_emocao{emocao}.png'
            caminho_imagem = os.path.join(raiz_projeto, nome_arquivo)
            img.save(caminho_imagem)
            
            print(f"✅ Imagem salva com sucesso em: {caminho_imagem}\n")
            # ========================================================
            
            break
            
            
            
    except Exception as e:
        print(f"\nOcorreu um erro durante a execução: {e}")

if __name__ == "__main__":
    main()
