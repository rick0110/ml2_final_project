#!/usr/bin/env python3
"""
Testes unitários para validar dimensões e fluxo de tensores 
dos módulos GST e das Loss Functions.
"""

import sys
import unittest
from pathlib import Path
import torch

# Adicionar a raiz do projeto ao path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Tente importar os módulos (ajuste os caminhos se necessário)
try:
    from models.GST import GST
    from losses import CombinedTTSLoss
except ImportError as e:
    print(f"Erro de importação. Certifique-se de que os caminhos estão corretos: {e}")
    sys.exit(1)


class TestTTSArchitecture(unittest.TestCase):
    def setUp(self):
        # Hiperparâmetros simulados
        self.batch_size = 4
        self.n_mels = 80
        self.time_steps = 150
        self.hidden_size = 256
        self.num_tokens = 10
        self.device = torch.device("cpu")

        # Configurar reprodutibilidade
        torch.manual_seed(42)

    def test_gst_dimensions_and_flow(self):
        """Testa se o GST processa o mel spectrogram e retorna as dimensões corretas."""
        print("\n--- Testando GST ---")
        model = GST(
            n_conv_layers=6,
            hidden_size=self.hidden_size,
            n_style_tokens=self.num_tokens,
            n_mels=self.n_mels,
            n_heads=4
        ).to(self.device)

        # Simular Mel Spectrogram [Batch, n_mels, Time]
        dummy_mel = torch.randn(self.batch_size, 1, self.n_mels, self.time_steps).to(self.device)        
        # Forward pass
        style_embedding = model(dummy_mel)

        # Verificações de dimensão
        self.assertEqual(
            style_embedding.shape, 
            (self.batch_size, self.hidden_size),
            f"Erro na dimensão do output. Esperado: {(self.batch_size, self.hidden_size)}, Recebido: {style_embedding.shape}"
        )
        
        self.assertEqual(
            model.style_tokens.shape,
            (self.num_tokens, self.hidden_size),
            f"Erro na dimensão da matriz global de tokens. Esperado: {(self.num_tokens, self.hidden_size)}, Recebido: {model.style_tokens.shape}"
        )
        print("✓ Dimensões do GST corretas!")

    def test_combined_loss_and_gradients(self):
        """Testa se a loss calcula corretamente e permite o fluxo de gradientes (backward)."""
        print("\n--- Testando Combined TTS Loss ---")
        criterion = CombinedTTSLoss(
            weight_reconstruction=1.0,
            weight_diversity=0.5,
            diversity_margin=0.1
        ).to(self.device)

        # Simular outputs do modelo
        predicted_mel = torch.randn(self.batch_size, self.n_mels, self.time_steps, requires_grad=True).to(self.device)
        target_mel = torch.randn(self.batch_size, self.n_mels, self.time_steps).to(self.device)
        
        # Simular a matriz global de tokens [num_tokens, dim]
        global_style_tokens = torch.randn(self.num_tokens, self.hidden_size, requires_grad=True).to(self.device)

        # Calcular Loss
        total_loss, recon_loss, div_loss = criterion(
            predicted_mel=predicted_mel,
            target_mel=target_mel,
            global_style_tokens=global_style_tokens
        )

        # Verificações de tipo e dimensão (devem ser escalares)
        self.assertTrue(total_loss.dim() == 0, "Total loss não é um escalar.")
        self.assertTrue(recon_loss.dim() == 0, "Recon loss não é um escalar.")
        self.assertTrue(div_loss.dim() == 0, "Div loss não é um escalar.")

        # Testar Backward Pass (Fluxo de gradientes)
        try:
            total_loss.backward()
            gradients_flow = predicted_mel.grad is not None and global_style_tokens.grad is not None
            self.assertTrue(gradients_flow, "Os gradientes não estão a fluir para os tensores de entrada.")
            print("✓ Loss functions e propagação de gradientes corretas!")
        except Exception as e:
            self.fail(f"O backward pass falhou com o erro: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)