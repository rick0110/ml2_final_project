#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Adicionando os caminhos como strings para garantir compatibilidade no sys.path
sys.path.insert(0, str(PROJECT_ROOT / "src" / "training" / "training-gst-only"))  
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))

from GST import GST
from TTS_GST import TTS_GST

from losses import CombinedTTSLoss
from interpretability_utils import (
    log_gst_attention_heatmap,
    log_style_token_similarity,
    log_style_token_embeddings
)


class TestTTSArchitecture(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.n_mels = 80
        self.time_steps = 150
        self.hidden_size = 256
        self.num_tokens = 10
        self.device = torch.device("cpu")
        torch.manual_seed(42)

        self.dummy_audio_mel = torch.randn(self.batch_size, 1, self.n_mels, self.time_steps).to(self.device)
        self.dummy_text_embed = torch.randn(self.batch_size, self.hidden_size, 50).to(self.device)

        self.mock_tb_logger = MagicMock()
        self.mock_tb_logger.writer = MagicMock()

    def test_gst_dimensions_and_flow(self):
        model = GST(
            n_conv_layers=6,
            hidden_size=self.hidden_size,
            n_style_tokens=self.num_tokens,
            n_mels=self.n_mels,
            n_heads=4
        ).to(self.device)

        # AQUI: Pedimos explicitamente as atenções no teste (return_att_weights=True)
        style_embedding, att_weights = model(self.dummy_audio_mel, return_att_weights=True)

        self.assertEqual(
            style_embedding.shape, 
            (self.batch_size, 1, self.hidden_size),
            f"Expected embedding shape {(self.batch_size, self.hidden_size)}, got {style_embedding.shape}"
        )
        
        self.assertEqual(
            att_weights.shape,
            (self.batch_size, self.num_tokens),
            f"Expected attention weights shape {(self.batch_size, self.num_tokens)}, got {att_weights.shape}"
        )

        self.assertEqual(
            model.style_tokens.shape,
            (self.num_tokens, self.hidden_size),
            f"Expected global tokens matrix shape {(self.num_tokens, self.hidden_size)}, got {model.style_tokens.shape}"
        )

        self.assertFalse(torch.isnan(style_embedding).any(), "Style embedding contains NaNs.")
        self.assertFalse(torch.isnan(att_weights).any(), "Attention weights contain NaNs.")

    def test_combined_loss_and_gradients(self):
        criterion = CombinedTTSLoss(
            weight_reconstruction=1.0,
            weight_diversity=0.5,
            diversity_margin=0.1
        ).to(self.device)

        predicted_mel = torch.randn(self.batch_size, self.n_mels, self.time_steps, requires_grad=True).to(self.device)
        target_mel = torch.randn(self.batch_size, self.n_mels, self.time_steps).to(self.device)
        global_style_tokens = torch.randn(self.num_tokens, self.hidden_size, requires_grad=True).to(self.device)

        total_loss, recon_loss, div_loss = criterion(
            predicted_mel=predicted_mel,
            target_mel=target_mel,
            global_style_tokens=global_style_tokens
        )

        self.assertEqual(total_loss.dim(), 0, "Total loss is not a scalar.")
        self.assertEqual(recon_loss.dim(), 0, "Recon loss is not a scalar.")
        self.assertEqual(div_loss.dim(), 0, "Div loss is not a scalar.")

        try:
            total_loss.backward()
            gradients_flow = predicted_mel.grad is not None and global_style_tokens.grad is not None
            self.assertTrue(gradients_flow, "Gradients are not flowing back to the input tensors.")
        except Exception as e:
            self.fail(f"Backward pass failed with error: {e}")

    # AQUI ESTÁ A CORREÇÃO: Usando a string normal de patch que funciona com a sua injeção de sys.path
    @patch("TTS_GST.load_hifigan_model")
    def test_tts_gst_forward(self, mock_load_hifigan):
        mock_spec_generator = MagicMock()
        mock_vocoder = MagicMock()
        
        mock_spec_generator.return_value = torch.randn(self.batch_size, 80, 50) 
        mock_vocoder.return_value = torch.randn(self.batch_size, 1, 16000)
        
        mock_load_hifigan.return_value = (mock_spec_generator, mock_vocoder)
        
        model = TTS_GST(num_gst_tokens=self.num_tokens, gst_token_dim=self.hidden_size).to(self.device)
        
        audio_output, att_weights, _ = model(self.dummy_text_embed, self.dummy_audio_mel, return_att_weights=True)
        
        self.assertTrue(mock_spec_generator.called, "Spec Generator was not called.")
        self.assertTrue(mock_vocoder.called, "Vocoder was not called.")
        
        self.assertEqual(audio_output.shape, (self.batch_size, 1, 16000), "Incorrect audio output shape.")
        self.assertEqual(att_weights.shape, (self.batch_size, self.num_tokens), "Attention weights shape is incorrect.")

    def test_log_gst_attention_heatmap(self):
        dummy_att_weights = torch.rand(self.batch_size, self.num_tokens)
        epoch = 1
        
        try:
            log_gst_attention_heatmap(self.mock_tb_logger, dummy_att_weights, epoch)
            self.mock_tb_logger.writer.add_figure.assert_called_once()
        except Exception as e:
            self.fail(f"log_gst_attention_heatmap failed with exception: {e}")

    def test_log_style_token_similarity(self):
        gst_mock = MagicMock()
        gst_mock.style_tokens = torch.randn(self.num_tokens, self.hidden_size).to(self.device)
        epoch = 1
        
        try:
            log_style_token_similarity(self.mock_tb_logger, gst_mock, epoch)
            self.mock_tb_logger.writer.add_figure.assert_called_once()
        except Exception as e:
            self.fail(f"log_style_token_similarity failed with exception: {e}")

    def test_log_style_token_embeddings(self):
        gst_mock = MagicMock()
        gst_mock.style_tokens = torch.randn(self.num_tokens, self.hidden_size).to(self.device)
        epoch = 1
        
        try:
            log_style_token_embeddings(self.mock_tb_logger, gst_mock, epoch)
            self.mock_tb_logger.writer.add_embedding.assert_called_once()
            
            call_args = self.mock_tb_logger.writer.add_embedding.call_args[1]
            self.assertEqual(len(call_args['metadata']), self.num_tokens, "Incorrect token metadata length.")
            self.assertEqual(call_args['tag'], "GST_Tokens_Latent_Space")
        except Exception as e:
            self.fail(f"log_style_token_embeddings failed with exception: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)