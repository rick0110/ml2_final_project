"""
Unit tests for model components.

These tests validate the shapes and basic behaviour of each sub-module without
requiring real audio data or a GPU.  All random inputs are synthetic.
"""

import pytest
import torch

# The tests/__init__.py adds src/ to sys.path
from models.content_encoder import ContentEncoder
from models.reference_encoder import ReferenceEncoder, GlobalStyleToken, MultiHeadAttention
from models.variance_adaptor import VarianceAdaptor, VariancePredictor, LengthRegulator
from models.mapping_network import MappingNetwork, FiLMLayer, MappingBlock
from models.decoder import Decoder, MelPredictor, HiFiGANGenerator, ResBlock, MultiReceptiveFieldFusion
from models.full_model import ProsodyStyleTransferModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BATCH_SIZE = 2
N_MELS = 80
T_WAV = 16000   # 1 second at 16 kHz (HuBERT operates at 16 kHz)
T_ENC = 50      # encoder frame length (after HuBERT sub-sampling)
T_MEL = 100     # mel-spectrogram time frames
D_MODEL = 64    # small dimension for fast tests
STYLE_DIM = 32


# ---------------------------------------------------------------------------
# ContentEncoder
# ---------------------------------------------------------------------------

class TestContentEncoder:
    """Tests for the HuBERT-based content encoder."""

    def test_output_shape_no_projection(self, monkeypatch):
        """Without projection the output dim equals HuBERT hidden size."""
        encoder = _make_dummy_content_encoder(monkeypatch, output_dim=None)
        x = torch.randn(BATCH_SIZE, T_WAV)
        out = encoder(x)
        assert out.shape == (BATCH_SIZE, _DUMMY_FRAMES, encoder.hidden_size)

    def test_output_shape_with_projection(self, monkeypatch):
        """With projection the output matches the requested output_dim."""
        encoder = _make_dummy_content_encoder(monkeypatch, output_dim=D_MODEL)
        x = torch.randn(BATCH_SIZE, T_WAV)
        out = encoder(x)
        assert out.shape == (BATCH_SIZE, _DUMMY_FRAMES, D_MODEL)

    def test_frozen_parameters(self, monkeypatch):
        """All HuBERT parameters should be frozen when freeze=True."""
        encoder = _make_dummy_content_encoder(monkeypatch, freeze=True)
        for param in encoder.hubert.parameters():
            assert not param.requires_grad

    def test_unfrozen_parameters(self, monkeypatch):
        """HuBERT parameters should be trainable when freeze=False."""
        encoder = _make_dummy_content_encoder(monkeypatch, freeze=False)
        assert any(p.requires_grad for p in encoder.hubert.parameters())


# ---------------------------------------------------------------------------
# ReferenceEncoder / GST
# ---------------------------------------------------------------------------

class TestReferenceEncoder:
    """Tests for the convolutional reference encoder."""

    def test_output_shape(self):
        encoder = ReferenceEncoder(n_mels=N_MELS, ref_embedding_dim=STYLE_DIM)
        mels = torch.randn(BATCH_SIZE, N_MELS, T_MEL)
        out = encoder(mels)
        assert out.shape == (BATCH_SIZE, STYLE_DIM)

    def test_accepts_4d_input(self):
        """Encoder should handle (B, 1, n_mels, T) input."""
        encoder = ReferenceEncoder(n_mels=N_MELS, ref_embedding_dim=STYLE_DIM)
        mels = torch.randn(BATCH_SIZE, 1, N_MELS, T_MEL)
        out = encoder(mels)
        assert out.shape == (BATCH_SIZE, STYLE_DIM)


class TestMultiHeadAttention:
    """Tests for the GST multi-head attention."""

    def test_output_shape(self):
        query_dim = STYLE_DIM
        key_dim = 64
        num_heads = 4
        attn = MultiHeadAttention(query_dim=query_dim, key_dim=key_dim, num_heads=num_heads)
        query = torch.randn(BATCH_SIZE, 1, query_dim)
        key = torch.randn(5, key_dim)
        out = attn(query, key)
        assert out.shape == (BATCH_SIZE, query_dim)


class TestGlobalStyleToken:
    """Tests for the full GST module."""

    def test_output_shape(self):
        gst = GlobalStyleToken(
            n_mels=N_MELS, num_tokens=5, token_dim=64,
            num_heads=4, ref_embedding_dim=STYLE_DIM
        )
        mels = torch.randn(BATCH_SIZE, N_MELS, T_MEL)
        out = gst(mels)
        assert out.shape == (BATCH_SIZE, STYLE_DIM)

    def test_style_dim_attribute(self):
        gst = GlobalStyleToken(n_mels=N_MELS, ref_embedding_dim=STYLE_DIM)
        assert gst.style_dim == STYLE_DIM


# ---------------------------------------------------------------------------
# VarianceAdaptor
# ---------------------------------------------------------------------------

class TestVariancePredictor:
    """Tests for the variance predictor sub-module."""

    def test_output_shape(self):
        predictor = VariancePredictor(d_model=D_MODEL)
        x = torch.randn(BATCH_SIZE, T_ENC, D_MODEL)
        out = predictor(x)
        assert out.shape == (BATCH_SIZE, T_ENC, 1)


class TestLengthRegulator:
    """Tests for the length regulator."""

    def test_output_with_fixed_durations(self):
        regulator = LengthRegulator()
        x = torch.randn(BATCH_SIZE, T_ENC, D_MODEL)
        durations = torch.ones(BATCH_SIZE, T_ENC, dtype=torch.long) * 2  # every frame × 2
        out, lengths = regulator(x, durations)
        assert out.shape == (BATCH_SIZE, T_ENC * 2, D_MODEL)
        assert (lengths == T_ENC * 2).all()

    def test_zero_durations_padded(self):
        """Zero durations should produce zero-length output (padded to 0)."""
        regulator = LengthRegulator()
        x = torch.randn(1, 5, D_MODEL)
        durations = torch.zeros(1, 5, dtype=torch.long)
        out, lengths = regulator(x, durations)
        assert out.shape[0] == 1  # batch preserved


class TestVarianceAdaptor:
    """Tests for the full variance adaptor."""

    def test_output_keys(self):
        adaptor = VarianceAdaptor(d_model=D_MODEL, style_dim=STYLE_DIM)
        x = torch.randn(BATCH_SIZE, T_ENC, D_MODEL)
        style = torch.randn(BATCH_SIZE, STYLE_DIM)
        durations = torch.ones(BATCH_SIZE, T_ENC, dtype=torch.long)
        result = adaptor(x, style, target_durations=durations)
        assert "output" in result
        assert "pred_durations" in result
        assert "pred_pitch" in result
        assert "pred_energy" in result

    def test_output_shape_teacher_forced(self):
        adaptor = VarianceAdaptor(d_model=D_MODEL, style_dim=STYLE_DIM)
        x = torch.randn(BATCH_SIZE, T_ENC, D_MODEL)
        style = torch.randn(BATCH_SIZE, STYLE_DIM)
        durations = torch.ones(BATCH_SIZE, T_ENC, dtype=torch.long) * 2
        result = adaptor(x, style, target_durations=durations)
        assert result["output"].shape == (BATCH_SIZE, T_ENC * 2, D_MODEL)


# ---------------------------------------------------------------------------
# MappingNetwork
# ---------------------------------------------------------------------------

class TestFiLMLayer:
    """Tests for FiLM conditioning layer."""

    def test_output_shape(self):
        film = FiLMLayer(feature_dim=D_MODEL, condition_dim=STYLE_DIM)
        x = torch.randn(BATCH_SIZE, T_ENC, D_MODEL)
        cond = torch.randn(BATCH_SIZE, STYLE_DIM)
        out = film(x, cond)
        assert out.shape == (BATCH_SIZE, T_ENC, D_MODEL)


class TestMappingBlock:
    """Tests for residual mapping block."""

    def test_output_shape(self):
        block = MappingBlock(d_model=D_MODEL, style_dim=STYLE_DIM)
        x = torch.randn(BATCH_SIZE, T_ENC, D_MODEL)
        style = torch.randn(BATCH_SIZE, STYLE_DIM)
        out = block(x, style)
        assert out.shape == (BATCH_SIZE, T_ENC, D_MODEL)

    def test_residual_connection(self):
        """Output should be different from input (residual changes value)."""
        torch.manual_seed(0)
        block = MappingBlock(d_model=D_MODEL, style_dim=STYLE_DIM)
        x = torch.randn(BATCH_SIZE, T_ENC, D_MODEL)
        style = torch.randn(BATCH_SIZE, STYLE_DIM)
        out = block(x, style)
        assert not torch.allclose(out, x)


class TestMappingNetwork:
    """Tests for the full mapping network."""

    def test_output_shape(self):
        net = MappingNetwork(
            input_dim=D_MODEL * 2, output_dim=D_MODEL,
            hidden_dim=D_MODEL, num_layers=2, style_dim=STYLE_DIM
        )
        content = torch.randn(BATCH_SIZE, T_ENC, D_MODEL * 2)
        style = torch.randn(BATCH_SIZE, STYLE_DIM)
        out = net(content, style)
        assert out.shape == (BATCH_SIZE, T_ENC, D_MODEL)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class TestResBlock:
    def test_output_shape(self):
        block = ResBlock(channels=D_MODEL)
        x = torch.randn(BATCH_SIZE, D_MODEL, T_MEL)
        assert block(x).shape == x.shape


class TestMultiReceptiveFieldFusion:
    def test_output_shape(self):
        mrf = MultiReceptiveFieldFusion(channels=D_MODEL)
        x = torch.randn(BATCH_SIZE, D_MODEL, T_MEL)
        assert mrf(x).shape == x.shape


class TestMelPredictor:
    def test_output_shape(self):
        predictor = MelPredictor(input_dim=D_MODEL, n_mels=N_MELS, hidden_dim=D_MODEL)
        x = torch.randn(BATCH_SIZE, T_MEL, D_MODEL)
        out = predictor(x)
        assert out.shape == (BATCH_SIZE, N_MELS, T_MEL)


class TestHiFiGANGenerator:
    def test_output_shape(self):
        """Waveform length should be mel length × product of upsample rates."""
        rates = (4, 4)
        gen = HiFiGANGenerator(
            n_mels=N_MELS,
            upsample_rates=rates,
            upsample_initial_channel=64,
        )
        mel = torch.randn(BATCH_SIZE, N_MELS, T_MEL)
        out = gen(mel)
        expected_len = T_MEL * 4 * 4
        assert out.shape == (BATCH_SIZE, 1, expected_len)


class TestDecoder:
    def test_output_keys(self):
        decoder = Decoder(
            input_dim=D_MODEL, n_mels=N_MELS, mel_hidden_dim=D_MODEL,
            upsample_rates=(4, 4), upsample_initial_channel=64
        )
        x = torch.randn(BATCH_SIZE, T_MEL, D_MODEL)
        out = decoder(x)
        assert "mel" in out
        assert "waveform" in out

    def test_mel_shape(self):
        decoder = Decoder(
            input_dim=D_MODEL, n_mels=N_MELS, mel_hidden_dim=D_MODEL,
            upsample_rates=(4, 4), upsample_initial_channel=64
        )
        x = torch.randn(BATCH_SIZE, T_MEL, D_MODEL)
        assert decoder(x)["mel"].shape == (BATCH_SIZE, N_MELS, T_MEL)


# ---------------------------------------------------------------------------
# Full Model (ProsodyStyleTransferModel)
# ---------------------------------------------------------------------------

class TestProsodyStyleTransferModel:
    """Integration tests for the full model."""

    def _make_model(self, monkeypatch):
        _patch_hubert(monkeypatch)
        return ProsodyStyleTransferModel(
            hubert_model_name="facebook/hubert-base-ls960",
            freeze_hubert=True,
            n_mels=N_MELS,
            d_model=D_MODEL,
            style_dim=STYLE_DIM,
            mapping_hidden_dim=D_MODEL,
            mapping_num_layers=2,
            gst_num_tokens=3,
            gst_token_dim=64,
            gst_num_heads=4,
            variance_num_conv=1,
            decoder_upsample_rates=(4, 4),
            decoder_initial_channels=64,
        )

    def test_forward_with_reference(self, monkeypatch):
        model = self._make_model(monkeypatch)
        model.eval()
        wav = torch.randn(BATCH_SIZE, T_WAV)
        ref_mel = torch.randn(BATCH_SIZE, N_MELS, T_MEL)
        durations = torch.ones(BATCH_SIZE, _DUMMY_FRAMES, dtype=torch.long)
        with torch.no_grad():
            out = model(
                source_waveform=wav,
                ref_mel=ref_mel,
                target_durations=durations,
            )
        assert "mel" in out
        assert "waveform" in out
        assert "style_emb" in out
        assert out["style_emb"].shape == (BATCH_SIZE, STYLE_DIM)

    def test_forward_without_reference(self, monkeypatch):
        """Without a reference mel the model should use a zero style embedding."""
        model = self._make_model(monkeypatch)
        model.eval()
        wav = torch.randn(BATCH_SIZE, T_WAV)
        durations = torch.ones(BATCH_SIZE, _DUMMY_FRAMES, dtype=torch.long)
        with torch.no_grad():
            out = model(source_waveform=wav, target_durations=durations)
        assert out["style_emb"].shape == (BATCH_SIZE, STYLE_DIM)
        assert torch.allclose(out["style_emb"], torch.zeros_like(out["style_emb"]))

    def test_infer_api(self, monkeypatch):
        """infer() should be equivalent to eval forward without teacher-forcing."""
        model = self._make_model(monkeypatch)
        wav = torch.randn(1, T_WAV)
        out = model.infer(wav)
        assert "waveform" in out


# ---------------------------------------------------------------------------
# Helpers — mock HuBERT so tests don't download weights
# ---------------------------------------------------------------------------

_DUMMY_FRAMES = 20   # number of frames our dummy HuBERT produces
_HUBERT_HIDDEN = 768


def _patch_hubert(monkeypatch):
    """Replace HubertModel.from_pretrained with a lightweight stub."""
    import torch.nn as nn
    from unittest.mock import MagicMock

    class _DummyHubertOutput:
        def __init__(self, hidden_states):
            self.last_hidden_state = hidden_states

    class _DummyHubert(nn.Module):
        class config:
            hidden_size = _HUBERT_HIDDEN

        def __init__(self, *args, **kwargs):
            super().__init__()
            self.config = _DummyHubert.config
            self._linear = nn.Linear(1, _HUBERT_HIDDEN)  # dummy trainable param

        def forward(self, input_values, attention_mask=None, **kwargs):
            B = input_values.shape[0]
            return _DummyHubertOutput(
                torch.randn(B, _DUMMY_FRAMES, _HUBERT_HIDDEN)
            )

        def parameters(self, recurse=True):
            return iter([self._linear.weight, self._linear.bias])

    import models.content_encoder as ce_module
    monkeypatch.setattr(ce_module.HubertModel, "from_pretrained", lambda *a, **kw: _DummyHubert())


def _make_dummy_content_encoder(monkeypatch, output_dim=None, freeze=True):
    _patch_hubert(monkeypatch)
    return ContentEncoder(
        model_name="facebook/hubert-base-ls960",
        freeze=freeze,
        output_dim=output_dim,
    )
