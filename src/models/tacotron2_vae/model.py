"""
Tacotron2 VAE Model implementation.

This module defines the Tacotron2 VAE model, which combines a
Tacotron 2 architecture with a Variational Autoencoder (VAE) for
prosody modeling. It includes components for text embedding,
an encoder, a VAE-based prosody encoder (VAE_GST), a decoder with
attention, and a postnet.

Dependencies:
    - torch: PyTorch for neural network operations.
    - torch.nn: PyTorch neural network modules.
    - torch.nn.functional: PyTorch functional API.
    - math: For mathematical operations like sqrt.
    - models.tacotron2_vae.hparams: Hyperparameters for the model.
    - models.tacotron2_vae.layers: Custom linear and convolutional layers.
    - models.tacotron2_vae.modules: VAE_GST module for prosody encoding.
    - models.tacotron2_vae.utils: Utility functions for masks and device placement.

Typical Usage:
    >>> from src.models.tacotron2_vae.hparams import create_hparams
    >>> from src.models.tacotron2_vae.model import load_tacotron2_vae_model
    >>> hparams = create_hparams()
    >>> model = load_tacotron2_vae_model(hparams)
    >>> # ... prepare inputs ...
    >>> outputs = model(inputs)
"""
from math import sqrt
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from models.tacotron2_vae.hparams import Tacotron2VAEHparams
from models.tacotron2_vae.layers import ConvNorm, LinearNorm
from models.tacotron2_vae.modules import VAE_GST
from models.tacotron2_vae.utils import get_mask_from_lengths, to_device

DROP_RATE: float = 0.5

from models.tacotron2_vae.hparams import Tacotron2VAEHparams
from models.tacotron2_vae.layers import ConvNorm, LinearNorm
from models.tacotron2_vae.modules import VAE_GST
from models.tacotron2_vae.utils import get_mask_from_lengths, to_device

DROP_RATE: float = 0.5


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size, attention_dim):
        super().__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(
            2,
            attention_n_filters,
            kernel_size=attention_kernel_size,
            padding=padding,
            bias=False,
            stride=1,
            dilation=1,
        )
        self.location_dense = LinearNorm(
            attention_n_filters, attention_dim, bias=False, w_init_gain="tanh"
        )

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(
        self,
        attention_rnn_dim,
        embedding_dim,
        attention_dim,
        attention_location_n_filters,
        attention_location_kernel_size,
    ):
        super().__init__()
        self.query_layer = LinearNorm(
            attention_rnn_dim, attention_dim, bias=False, w_init_gain="tanh"
        )
        self.memory_layer = LinearNorm(
            embedding_dim, attention_dim, bias=False, w_init_gain="tanh"
        )
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(
            attention_location_n_filters,
            attention_location_kernel_size,
            attention_dim,
        )
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory, attention_weights_cat):
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(
            torch.tanh(processed_query + processed_attention_weights + processed_memory)
        )
        return energies.squeeze(-1)

    def forward(self, attention_hidden_state, memory, processed_memory, attention_weights_cat, mask):
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat
        )
        if mask is not None:
            alignment = alignment.masked_fill(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        return attention_context.squeeze(1), attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super().__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False) for in_size, out_size in zip(in_sizes, sizes)]
        )

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=DROP_RATE, training=self.training)
        return x


class Postnet(nn.Module):
    def __init__(self, hparams: Tacotron2VAEHparams):
        super().__init__()
        self.convolutions = nn.ModuleList()
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    hparams.n_mel_channels,
                    hparams.postnet_embedding_dim,
                    kernel_size=hparams.postnet_kernel_size,
                    stride=1,
                    padding=int((hparams.postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="tanh",
                ),
                nn.BatchNorm1d(hparams.postnet_embedding_dim),
            )
        )
        for _ in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        hparams.postnet_embedding_dim,
                        hparams.postnet_embedding_dim,
                        kernel_size=hparams.postnet_kernel_size,
                        stride=1,
                        padding=int((hparams.postnet_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain="tanh",
                    ),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim),
                )
            )
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    hparams.postnet_embedding_dim,
                    hparams.n_mel_channels,
                    kernel_size=hparams.postnet_kernel_size,
                    stride=1,
                    padding=int((hparams.postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="linear",
                ),
                nn.BatchNorm1d(hparams.n_mel_channels),
            )
        )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), DROP_RATE, self.training)
        x = F.dropout(self.convolutions[-1](x), DROP_RATE, self.training)
        return x


class Encoder(nn.Module):
    def __init__(self, hparams: Tacotron2VAEHparams):
        super().__init__()
        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(
                    hparams.encoder_embedding_dim,
                    hparams.encoder_embedding_dim,
                    kernel_size=hparams.encoder_kernel_size,
                    stride=1,
                    padding=int((hparams.encoder_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="relu",
                ),
                nn.BatchNorm1d(hparams.encoder_embedding_dim),
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        self.lstm = nn.LSTM(
            hparams.encoder_embedding_dim,
            int(hparams.encoder_embedding_dim / 2),
            1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x, input_lengths):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), DROP_RATE, self.training)
        x = x.transpose(1, 2)
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), DROP_RATE, self.training)
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        return outputs


class Decoder(nn.Module):
    def __init__(self, hparams: Tacotron2VAEHparams):
        super().__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout

        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,
            [hparams.prenet_dim, hparams.prenet_dim],
        )
        self.attention_rnn = nn.LSTMCell(
            hparams.prenet_dim + self.encoder_embedding_dim,
            hparams.attention_rnn_dim,
        )
        self.attention_layer = Attention(
            hparams.attention_rnn_dim,
            self.encoder_embedding_dim,
            hparams.attention_dim,
            hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size,
        )
        self.decoder_rnn = nn.LSTMCell(
            hparams.attention_rnn_dim + self.encoder_embedding_dim,
            hparams.decoder_rnn_dim,
            1,
        )
        self.linear_projection = LinearNorm(
            hparams.decoder_rnn_dim + self.encoder_embedding_dim,
            hparams.n_mel_channels * hparams.n_frames_per_step,
        )
        self.gate_layer = LinearNorm(
            hparams.decoder_rnn_dim + self.encoder_embedding_dim,
            1,
            bias=True,
            w_init_gain="sigmoid",
        )

    def get_go_frame(self, memory):
        batch_size = memory.size(0)
        return memory.new_zeros(batch_size, self.n_mel_channels * self.n_frames_per_step)

    def initialize_decoder_states(self, memory, mask):
        batch_size = memory.size(0)
        max_time = memory.size(1)

        self.attention_hidden = memory.new_zeros(batch_size, self.attention_rnn_dim)
        self.attention_cell = memory.new_zeros(batch_size, self.attention_rnn_dim)
        self.decoder_hidden = memory.new_zeros(batch_size, self.decoder_rnn_dim)
        self.decoder_cell = memory.new_zeros(batch_size, self.decoder_rnn_dim)
        self.attention_weights = memory.new_zeros(batch_size, max_time)
        self.attention_weights_cum = memory.new_zeros(batch_size, max_time)
        self.attention_context = memory.new_zeros(batch_size, self.encoder_embedding_dim)

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1) / self.n_frames_per_step),
            -1,
        )
        return decoder_inputs.transpose(0, 1)

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        alignments = torch.stack(alignments).transpose(0, 1)
        gate_outputs = torch.stack(gate_outputs)
        if len(gate_outputs.size()) == 1:
            gate_outputs = gate_outputs.unsqueeze(1)
        gate_outputs = gate_outputs.transpose(0, 1).contiguous()
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        mel_outputs = mel_outputs.view(mel_outputs.size(0), -1, self.n_mel_channels)
        mel_outputs = mel_outputs.transpose(1, 2)
        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell)
        )
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training
        )
        self.attention_cell = F.dropout(
            self.attention_cell, self.p_attention_dropout, self.training
        )

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1), self.attention_weights_cum.unsqueeze(1)),
            dim=1,
        )
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden,
            self.memory,
            self.processed_memory,
            attention_weights_cat,
            self.mask,
        )
        self.attention_weights_cum += self.attention_weights

        decoder_input = torch.cat((self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell)
        )
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training
        )
        self.decoder_cell = F.dropout(
            self.decoder_cell, self.p_decoder_dropout, self.training
        )

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1
        )
        decoder_output = self.linear_projection(decoder_hidden_attention_context)
        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths):
        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(memory, mask=~get_mask_from_lengths(memory_lengths))

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(decoder_input)
            mel_outputs.append(mel_output.squeeze(1))
            gate_outputs.append(gate_output.squeeze())
            alignments.append(attention_weights)

        return self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments)

    def inference(self, memory):
        decoder_input = self.get_go_frame(memory)
        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)
            mel_outputs.append(mel_output.squeeze(1))
            gate_outputs.append(gate_output)
            alignments.append(alignment)

            if torch.sigmoid(gate_output) > self.gate_threshold:
                break
            if len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break
            decoder_input = mel_output

        return self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments)


class Tacotron2(nn.Module):
    def __init__(self, hparams: Tacotron2VAEHparams):
        super().__init__()
        self.hparams = hparams
        self.mask_padding = hparams.mask_padding
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step

        self.transcript_embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim
        )

        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std
        self.transcript_embedding.weight.data.uniform_(-val, val)

        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)
        self.vae_gst = VAE_GST(hparams)

    def parse_batch(self, batch, device: torch.device):
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths, _ = batch
        text_padded = to_device(text_padded, device).long()
        input_lengths = to_device(input_lengths, device).long()
        mel_padded = to_device(mel_padded, device).float()
        gate_padded = to_device(gate_padded, device).float()
        output_lengths = to_device(output_lengths, device).long()

        return (
            text_padded,
            input_lengths,
            mel_padded,
            output_lengths,
        ), (mel_padded, gate_padded)

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)
            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)
        return outputs

    def forward(self, inputs):
        inputs, input_lengths, targets, output_lengths = inputs
        input_lengths = input_lengths
        output_lengths = output_lengths

        transcript_embedded_inputs = self.transcript_embedding(inputs).transpose(1, 2)
        transcript_outputs = self.encoder(transcript_embedded_inputs, input_lengths)

        prosody_outputs, mu, logvar, z = self.vae_gst(targets)
        prosody_outputs = prosody_outputs.unsqueeze(1).expand_as(transcript_outputs)
        encoder_outputs = transcript_outputs + prosody_outputs

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, targets, memory_lengths=input_lengths
        )
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments, mu, logvar, z],
            output_lengths,
        )


def load_tacotron2_vae_model(
    hparams: Tacotron2VAEHparams,
    device=None,
) -> Tacotron2:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Tacotron2(hparams).to(device)
    return model


def get_model_size_info(model: Tacotron2) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}
