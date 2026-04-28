"""Unit tests for training loss functions."""

import pytest
import torch

from training.losses import (
    MelLoss,
    DurationLoss,
    PitchLoss,
    EnergyLoss,
    TotalLoss,
)

B, N_MELS, T_ENC, T_DEC = 2, 80, 20, 40


class TestMelLoss:
    def test_identical_inputs_zero_loss(self):
        loss_fn = MelLoss()
        mel = torch.randn(B, N_MELS, T_DEC)
        assert loss_fn(mel, mel).item() == pytest.approx(0.0, abs=1e-6)

    def test_masked_loss(self):
        loss_fn = MelLoss()
        pred = torch.ones(B, N_MELS, T_DEC)
        target = torch.zeros(B, N_MELS, T_DEC)
        mask = torch.zeros(B, T_DEC)
        mask[:, :T_DEC // 2] = 1.0
        loss = loss_fn(pred, target, mask)
        assert loss.item() == pytest.approx(1.0, abs=1e-4)

    def test_output_is_scalar(self):
        loss_fn = MelLoss()
        loss = loss_fn(torch.randn(B, N_MELS, T_DEC), torch.randn(B, N_MELS, T_DEC))
        assert loss.shape == ()


class TestDurationLoss:
    def test_output_is_scalar(self):
        loss_fn = DurationLoss()
        pred = torch.randn(B, T_ENC, 1)
        target = torch.randint(1, 5, (B, T_ENC))
        assert loss_fn(pred, target).shape == ()

    def test_perfect_prediction_zero_loss(self):
        loss_fn = DurationLoss()
        durations = torch.ones(B, T_ENC, dtype=torch.long) * 3
        log_dur = torch.log(durations.float()).unsqueeze(-1)
        assert loss_fn(log_dur, durations).item() == pytest.approx(0.0, abs=1e-5)


class TestPitchLoss:
    def test_output_is_scalar(self):
        loss_fn = PitchLoss()
        pred = torch.randn(B, T_ENC, 1)
        target = torch.randn(B, T_ENC, 1)
        assert loss_fn(pred, target).shape == ()


class TestEnergyLoss:
    def test_output_is_scalar(self):
        loss_fn = EnergyLoss()
        pred = torch.randn(B, T_ENC, 1)
        target = torch.randn(B, T_ENC, 1)
        assert loss_fn(pred, target).shape == ()


class TestTotalLoss:
    def _dummy_outputs_and_targets(self):
        model_output = {
            "mel": torch.randn(B, N_MELS, T_DEC),
            "pred_durations": torch.randn(B, T_ENC, 1),
            "pred_pitch": torch.randn(B, T_ENC, 1),
            "pred_energy": torch.randn(B, T_ENC, 1),
        }
        targets = {
            "mel": torch.randn(B, N_MELS, T_DEC),
            "durations": torch.randint(1, 4, (B, T_ENC)),
            "pitch": torch.randn(B, T_ENC, 1),
            "energy": torch.randn(B, T_ENC, 1),
        }
        return model_output, targets

    def test_output_keys(self):
        loss_fn = TotalLoss()
        out, tgt = self._dummy_outputs_and_targets()
        losses = loss_fn(out, tgt)
        for key in ("total", "mel", "duration", "pitch", "energy"):
            assert key in losses

    def test_total_is_weighted_sum(self):
        weights = dict(mel_weight=2.0, duration_weight=0.5, pitch_weight=0.5, energy_weight=0.5)
        loss_fn = TotalLoss(**weights)
        out, tgt = self._dummy_outputs_and_targets()
        losses = loss_fn(out, tgt)
        expected = (
            2.0 * losses["mel"]
            + 0.5 * losses["duration"]
            + 0.5 * losses["pitch"]
            + 0.5 * losses["energy"]
        )
        assert torch.isclose(losses["total"], expected)
