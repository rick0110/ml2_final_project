"""
VITS-style Decoder.

This module implements the waveform decoder that converts frame-level acoustic
features into a raw waveform.  It is inspired by the decoder architecture of
VITS (Kim et al., "Conditional Variational Autoencoder with Adversarial
Learning for End-to-End Text-to-Speech", ICML 2021) and consists of:

    1. A mel-spectrogram predictor (linear + convolutional layers).
    2. A HiFi-GAN–style multi-receptive-field fusion (MRF) vocoder that
       up-samples the mel-spectrogram to a waveform.

For low-resource adaptation the mel predictor head can be fine-tuned on the
Portuguese corpus while the vocoder weights are reused from a large pre-trained
VITS checkpoint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# HiFi-GAN building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Residual block with dilated convolutions (from HiFi-GAN).

    Args:
        channels: Number of channels.
        kernel_size: Kernel size for the dilated convolutions.
        dilations: Sequence of dilation factors applied per sub-block.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 3, 5),
    ) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    dilation=d,
                    padding=self._same_pad(kernel_size, d),
                )
                for d in dilations
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2)
                for _ in dilations
            ]
        )

    @staticmethod
    def _same_pad(kernel_size: int, dilation: int) -> int:
        return (kernel_size - 1) * dilation // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            out = F.leaky_relu(x, 0.1)
            out = F.leaky_relu(c1(out), 0.1)
            out = c2(out)
            x = x + out
        return x


class MultiReceptiveFieldFusion(nn.Module):
    """Multi-receptive-field fusion (MRF) module from HiFi-GAN.

    Runs multiple residual blocks with different kernel sizes in parallel and
    sums their outputs.

    Args:
        channels: Number of input/output channels.
        resblock_kernel_sizes: Kernel sizes for each parallel residual block.
        resblock_dilations: Dilation patterns (one per resblock).
    """

    def __init__(
        self,
        channels: int,
        resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11),
        resblock_dilations: tuple[tuple[int, ...], ...] = (
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        ),
    ) -> None:
        super().__init__()
        assert len(resblock_kernel_sizes) == len(resblock_dilations)
        self.blocks = nn.ModuleList(
            [
                ResBlock(channels, k, d)
                for k, d in zip(resblock_kernel_sizes, resblock_dilations)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = sum(block(x) for block in self.blocks)
        return out / len(self.blocks)


# ---------------------------------------------------------------------------
# Mel predictor
# ---------------------------------------------------------------------------

class MelPredictor(nn.Module):
    """Predicts a mel-spectrogram from up-sampled frame features.

    Args:
        input_dim: Dimension of input frame features.
        n_mels: Number of mel-filter bins.
        hidden_dim: Intermediate convolution channels.
    """

    def __init__(
        self,
        input_dim: int = 256,
        n_mels: int = 80,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, n_mels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict mel-spectrogram.

        Args:
            x: Frame features, shape ``(B, T, input_dim)``.

        Returns:
            Mel-spectrogram, shape ``(B, n_mels, T)``.
        """
        return self.net(x.transpose(1, 2))  # (B, n_mels, T)


# ---------------------------------------------------------------------------
# HiFi-GAN vocoder
# ---------------------------------------------------------------------------

class HiFiGANGenerator(nn.Module):
    """Simplified HiFi-GAN generator that up-samples a mel-spectrogram to a waveform.

    Args:
        n_mels: Number of mel bins (input channels).
        upsample_rates: Up-sampling factors per transposed conv layer.
        upsample_initial_channel: Initial number of channels.
        resblock_kernel_sizes: Kernel sizes for MRF residual blocks.
        resblock_dilations: Dilation patterns for MRF residual blocks.
    """

    def __init__(
        self,
        n_mels: int = 80,
        upsample_rates: tuple[int, ...] = (8, 8, 2, 2),
        upsample_initial_channel: int = 512,
        resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11),
        resblock_dilations: tuple[tuple[int, ...], ...] = (
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        ),
    ) -> None:
        super().__init__()
        self.num_upsamples = len(upsample_rates)

        self.conv_pre = nn.Conv1d(n_mels, upsample_initial_channel, kernel_size=7, padding=3)

        self.ups = nn.ModuleList()
        self.mrfs = nn.ModuleList()
        channels = upsample_initial_channel
        for rate in upsample_rates:
            next_channels = channels // 2
            self.ups.append(
                nn.ConvTranspose1d(
                    channels,
                    next_channels,
                    kernel_size=rate * 2,
                    stride=rate,
                    padding=rate // 2,
                )
            )
            self.mrfs.append(
                MultiReceptiveFieldFusion(next_channels, resblock_kernel_sizes, resblock_dilations)
            )
            channels = next_channels

        self.conv_post = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv1d(channels, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Generate a waveform from a mel-spectrogram.

        Args:
            mel: Mel-spectrogram, shape ``(B, n_mels, T_mel)``.

        Returns:
            Raw waveform, shape ``(B, 1, T_wav)``.
        """
        x = self.conv_pre(mel)
        for up, mrf in zip(self.ups, self.mrfs):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            x = mrf(x)
        return self.conv_post(x)


# ---------------------------------------------------------------------------
# Combined Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """Full decoder: frame features → mel-spectrogram → waveform.

    Args:
        input_dim: Dimension of the input frame features from the variance adaptor.
        n_mels: Number of mel-spectrogram bins.
        mel_hidden_dim: Hidden dimension for the mel predictor.
        upsample_rates: Up-sampling rates for HiFi-GAN.
        upsample_initial_channel: Initial channels for HiFi-GAN.
    """

    def __init__(
        self,
        input_dim: int = 256,
        n_mels: int = 80,
        mel_hidden_dim: int = 256,
        upsample_rates: tuple[int, ...] = (8, 8, 2, 2),
        upsample_initial_channel: int = 512,
    ) -> None:
        super().__init__()
        self.mel_predictor = MelPredictor(input_dim, n_mels, mel_hidden_dim)
        self.vocoder = HiFiGANGenerator(
            n_mels=n_mels,
            upsample_rates=upsample_rates,
            upsample_initial_channel=upsample_initial_channel,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode frame features to waveform.

        Args:
            x: Frame features, shape ``(B, T, input_dim)``.

        Returns:
            Dictionary with:
                - ``"mel"``: Predicted mel-spectrogram ``(B, n_mels, T)``.
                - ``"waveform"``: Generated waveform ``(B, 1, T_wav)``.
        """
        mel = self.mel_predictor(x)       # (B, n_mels, T)
        waveform = self.vocoder(mel)      # (B, 1, T_wav)
        return {"mel": mel, "waveform": waveform}
