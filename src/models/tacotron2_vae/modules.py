import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tacotron2_vae.coord_conv import CoordConv2d
from models.tacotron2_vae.hparams import Tacotron2VAEHparams


class ReferenceEncoder(nn.Module):
    def __init__(self, hparams: Tacotron2VAEHparams):
        super().__init__()
        k = len(hparams.ref_enc_filters)
        filters = [1] + hparams.ref_enc_filters
        convs = [
            CoordConv2d(
                in_channels=filters[0],
                out_channels=filters[1],
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                with_r=True,
            )
        ]
        convs.extend(
            [
                nn.Conv2d(
                    in_channels=filters[i],
                    out_channels=filters[i + 1],
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                )
                for i in range(1, k)
            ]
        )
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features=hparams.ref_enc_filters[i]) for i in range(k)]
        )

        out_channels = self.calculate_channels(hparams.n_mel_channels, 3, 2, 1, k)
        self.gru = nn.GRU(
            input_size=hparams.ref_enc_filters[-1] * out_channels,
            hidden_size=hparams.E // 2,
            batch_first=True,
        )
        self.n_mels = hparams.n_mel_channels

    def forward(self, inputs):
        batch_size = inputs.size(0)
        out = inputs.contiguous().view(batch_size, 1, -1, self.n_mels)
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)

        out = out.transpose(1, 2)
        time_steps = out.size(1)
        out = out.contiguous().view(batch_size, time_steps, -1)
        _, out = self.gru(out)
        return out.squeeze(0)

    @staticmethod
    def calculate_channels(length, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            length = (length - kernel_size + 2 * pad) // stride + 1
        return length


class VAE_GST(nn.Module):
    def __init__(self, hparams: Tacotron2VAEHparams):
        super().__init__()
        self.ref_encoder = ReferenceEncoder(hparams)
        self.fc1 = nn.Linear(hparams.ref_enc_gru_size, hparams.z_latent_dim)
        self.fc2 = nn.Linear(hparams.ref_enc_gru_size, hparams.z_latent_dim)
        self.fc3 = nn.Linear(hparams.z_latent_dim, hparams.E)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        return mu

    def forward(self, inputs):
        enc_out = self.ref_encoder(inputs)
        mu = self.fc1(enc_out)
        logvar = self.fc2(enc_out)
        z = self.reparameterize(mu, logvar)
        style_embed = self.fc3(z)
        return style_embed, mu, logvar, z
