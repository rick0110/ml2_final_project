"""
Tacotron 2 VAE Model implementation.

Responsibilities:
    - Implement Tacotron2: Top-level module combining text encoder, VAE-GST, and attention-based decoder.
    - Implement Encoder: Multi-layer CNN and Bidirectional LSTM for text encoding.
    - Implement Decoder: LSTM-based autoregressive decoder with location-sensitive attention.
    - Implement Postnet: CNN-based refinement module for mel-spectrograms.

Main Classes:
    - Tacotron2: Main model integrating all components.
    - Encoder: Encodes phoneme sequences into embeddings.
    - Decoder: Autoregressively predicts mel-spectrogram frames.
    - Attention: Location-sensitive attention mechanism.
    - Prenet: Bottleneck layer for previous decoder output.
    - Postnet: Refines output mel-spectrogram.

Tensor Conventions:
    B = batch size
    T = sequence length (frames/tokens)
    n_mels = mel frequency bins
    H = hidden dimension
    L = latent dimension
"""
from math import sqrt
from typing import Any, Dict, List, Tuple, Union, Optional

import torch
from torch import Tensor
import torch.nn.functional as F
from torch import nn

try:
    from models.tacotron2_vae.hparams import Tacotron2VAEHparams
    from models.tacotron2_vae.layers import ConvNorm, LinearNorm
    from models.tacotron2_vae.modules import VAE_GST
    from models.tacotron2_vae.utils import get_mask_from_lengths, to_device
except ImportError:
    # Fallback for local imports
    from hparams import Tacotron2VAEHparams
    from layers import ConvNorm, LinearNorm
    from modules import VAE_GST
    from utils import get_mask_from_lengths, to_device

DROP_RATE: float = 0.5


class LocationLayer(nn.Module):
    """
    Location-sensitive attention layer.

    Architecture:
        Conv1d -> Linear

    Inputs:
        attention_weights_cat:
            Shape (B, 2, T_text)

    Outputs:
        processed_attention:
            Shape (B, T_text, attention_dim)
    """
    def __init__(self, attention_n_filters: int, attention_kernel_size: int, attention_dim: int) -> None:
        """
        Initialize the LocationLayer.

        Args:
            attention_n_filters (int): Number of convolution filters.
            attention_kernel_size (int): Convolution kernel size.
            attention_dim (int): Final output dimension.
        """
        super().__init__()
        padding: int = int((attention_kernel_size - 1) / 2)
        self.location_conv: ConvNorm = ConvNorm(
            2,
            attention_n_filters,
            kernel_size=attention_kernel_size,
            padding=padding,
            bias=False,
            stride=1,
            dilation=1,
        )
        self.location_dense: LinearNorm = LinearNorm(
            attention_n_filters, attention_dim, bias=False, w_init_gain="tanh"
        )

    def forward(self, attention_weights_cat: Tensor) -> Tensor:
        """
        Args:
            attention_weights_cat (Tensor): Concatenated weights (B, 2, T_text).
        
        Returns:
            Tensor: Processed weights (B, T_text, attention_dim).
        """
        processed_attention: Tensor = self.location_conv(attention_weights_cat) # (B, filters, T_text)
        processed_attention = processed_attention.transpose(1, 2)             # (B, T_text, filters)
        processed_attention = self.location_dense(processed_attention)       # (B, T_text, attention_dim)
        return processed_attention


class Attention(nn.Module):
    """
    Location-sensitive attention mechanism.

    Architecture:
        QueryLayer + MemoryLayer + LocationLayer -> Tanh -> V_Layer -> Softmax

    Inputs:
        attention_hidden_state: (B, attention_rnn_dim)
        memory: (B, T_text, encoder_dim)
        processed_memory: (B, T_text, attention_dim)
        attention_weights_cat: (B, 2, T_text)
        mask: (B, T_text)

    Outputs:
        attention_context: (B, encoder_dim)
        attention_weights: (B, T_text)
    """
    def __init__(
        self,
        attention_rnn_dim: int,
        embedding_dim: int,
        attention_dim: int,
        attention_location_n_filters: int,
        attention_location_kernel_size: int,
    ) -> None:
        """
        Initialize the Attention mechanism.

        Args:
            attention_rnn_dim (int): Hidden dimension of attention RNN.
            embedding_dim (int): Dimension of encoder output (memory).
            attention_dim (int): Internal attention dimension.
            attention_location_n_filters (int): Filters for location layer.
            attention_location_kernel_size (int): Kernel size for location layer.
        """
        super().__init__()
        self.query_layer: LinearNorm = LinearNorm(
            attention_rnn_dim, attention_dim, bias=False, w_init_gain="tanh"
        )
        self.memory_layer: LinearNorm = LinearNorm(
            embedding_dim, attention_dim, bias=False, w_init_gain="tanh"
        )
        self.v: LinearNorm = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer: LocationLayer = LocationLayer(
            attention_location_n_filters,
            attention_location_kernel_size,
            attention_dim,
        )
        self.ln: nn.LayerNorm = nn.LayerNorm(attention_dim)
        self.score_mask_value: float = -1e4

    def get_alignment_energies(self, query: Tensor, processed_memory: Tensor, attention_weights_cat: Tensor) -> Tensor:
        """
        Args:
            query (Tensor): (B, attention_rnn_dim)
            processed_memory (Tensor): (B, T_text, attention_dim)
            attention_weights_cat (Tensor): (B, 2, T_text)
        
        Returns:
            Tensor: Energies (B, T_text)
        """
        processed_query: Tensor = self.query_layer(query.unsqueeze(1)) # (B, 1, attention_dim)
        processed_attention_weights: Tensor = self.location_layer(attention_weights_cat) # (B, T_text, attention_dim)
        processed_attention_weights = self.ln(processed_attention_weights)
        energies: Tensor = self.v(
            torch.tanh(processed_query + processed_attention_weights + processed_memory)
        ) # (B, T_text, 1)
        return energies.squeeze(-1)

    def forward(
        self, 
        attention_hidden_state: Tensor, 
        memory: Tensor, 
        processed_memory: Tensor, 
        attention_weights_cat: Tensor, 
        mask: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of attention.

        Args:
            attention_hidden_state (Tensor): (B, attention_rnn_dim)
            memory (Tensor): (B, T_text, encoder_dim)
            processed_memory (Tensor): (B, T_text, attention_dim)
            attention_weights_cat (Tensor): (B, 2, T_text)
            mask (Tensor): (B, T_text)

        Returns:
            Tuple[Tensor, Tensor]: context (B, encoder_dim), weights (B, T_text).
        """
        alignment: Tensor = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat
        ) # (B, T_text)
        
        if mask is not None:
            alignment = alignment.masked_fill(mask, self.score_mask_value)

        attention_weights: Tensor = F.softmax(alignment, dim=1) # (B, T_text)
        attention_context: Tensor = torch.bmm(attention_weights.unsqueeze(1), memory) # (B, 1, encoder_dim)
        return attention_context.squeeze(1), attention_weights


class Prenet(nn.Module):
    """
    Bottleneck layer for decoder inputs.

    Architecture:
        [Linear -> ReLU -> Dropout] * L

    Inputs:
        x: (B, T_dec, input_dim)

    Outputs:
        x_bottleneck: (B, T_dec, sizes[-1])
    """
    def __init__(self, in_dim: int, sizes: List[int]) -> None:
        """
        Initialize Prenet.

        Args:
            in_dim (int): Input feature dimension.
            sizes (List[int]): Output dimensions for each layer.
        """
        super().__init__()
        in_sizes: List[int] = [in_dim] + sizes[:-1]
        self.layers: nn.ModuleList = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False) for in_size, out_size in zip(in_sizes, sizes)]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): (B, T_dec, in_dim) or (B, in_dim)
        
        Returns:
            Tensor: (B, T_dec, out_dim) or (B, out_dim)
        """
        for linear in self.layers:
            # Force pre-net dropout to remain active at inference time
            x = F.dropout(F.relu(linear(x)), p=DROP_RATE, training=True)
        return x


class Postnet(nn.Module):
    """
    Mel-spectrogram refinement module.

    Architecture:
        [Conv1d -> BN -> Tanh -> Dropout] * (N-1) -> Conv1d -> BN -> Dropout

    Inputs:
        x: (B, n_mels, T_mel)

    Outputs:
        residual: (B, n_mels, T_mel)
    """
    def __init__(self, hparams: Tacotron2VAEHparams) -> None:
        """
        Initialize Postnet.

        Args:
            hparams (Tacotron2VAEHparams): Model hyperparameters.
        """
        super().__init__()
        self.convolutions: nn.ModuleList = nn.ModuleList()
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

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): (B, n_mels, T_mel)
        
        Returns:
            Tensor: (B, n_mels, T_mel) residual to be added to input.
        """
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), DROP_RATE, self.training)
        x = F.dropout(self.convolutions[-1](x), DROP_RATE, self.training)
        return x


class Encoder(nn.Module):
    """
    Tacotron 2 Encoder.

    Architecture:
        [Conv1d -> BN -> ReLU -> Dropout] * 3 -> Bidirectional LSTM

    Inputs:
        x: (B, symbols_embedding_dim, T_text)
        input_lengths: (B,)

    Outputs:
        outputs: (B, T_text, encoder_embedding_dim)
    """
    def __init__(self, hparams: Tacotron2VAEHparams) -> None:
        """
        Initialize Encoder.

        Args:
            hparams (Tacotron2VAEHparams): Model hyperparameters.
        """
        super().__init__()
        convolutions: List[nn.Sequential] = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer: nn.Sequential = nn.Sequential(
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
        self.convolutions: nn.ModuleList = nn.ModuleList(convolutions)
        self.lstm: nn.LSTM = nn.LSTM(
            hparams.encoder_embedding_dim,
            int(hparams.encoder_embedding_dim / 2),
            1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: Tensor, input_lengths: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): (B, embed_dim, T_text)
            input_lengths (Tensor): (B,)
        
        Returns:
            Tensor: (B, T_text, encoder_dim)
        """
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), DROP_RATE, self.training) # (B, encoder_dim, T_text)
        
        x = x.transpose(1, 2) # (B, T_text, encoder_dim)
        
        # LSTM processing with packing
        input_lengths_cpu: Any = input_lengths.cpu().numpy()
        x_packed: Any = nn.utils.rnn.pack_padded_sequence(x, input_lengths_cpu, batch_first=True)
        self.lstm.flatten_parameters()
        outputs_packed, _ = self.lstm(x_packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs_packed, batch_first=True) # (B, T_text, encoder_dim)
        return outputs

    def inference(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): (B, embed_dim, T_text)
        
        Returns:
            Tensor: (B, T_text, encoder_dim)
        """
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), DROP_RATE, self.training)
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        return outputs


class Decoder(nn.Module):
    """
    Tacotron 2 Decoder.

    Architecture:
        Prenet -> AttentionRNN -> AttentionLayer -> DecoderRNN -> LinearProjection + GateLayer

    Inputs:
        memory: (B, T_text, encoder_dim)
        decoder_inputs: (B, n_mels, T_mel)
        memory_lengths: (B,)

    Outputs:
        mel_outputs: (B, n_mels, T_mel)
        gate_outputs: (B, T_mel)
        alignments: (B, T_mel, T_text)
    """
    def __init__(self, hparams: Tacotron2VAEHparams) -> None:
        """
        Initialize Decoder.

        Args:
            hparams (Tacotron2VAEHparams): Model hyperparameters.
        """
        super().__init__()
        self.n_mel_channels: int = hparams.n_mel_channels
        self.n_frames_per_step: int = hparams.n_frames_per_step
        self.encoder_embedding_dim: int = hparams.encoder_embedding_dim
        self.attention_rnn_dim: int = hparams.attention_rnn_dim
        self.decoder_rnn_dim: int = hparams.decoder_rnn_dim
        self.prenet_dim: int = hparams.prenet_dim
        self.max_decoder_steps: int = hparams.max_decoder_steps
        self.gate_threshold: float = hparams.gate_threshold
        self.p_attention_dropout: float = hparams.p_attention_dropout
        self.p_decoder_dropout: float = hparams.p_decoder_dropout
        self.p_scheduled_sampling: float = hparams.p_scheduled_sampling
        self.p_decoder_input_dropout: float = getattr(hparams, 'p_decoder_input_dropout', 0.5)

        self.prenet: Prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,
            [hparams.prenet_dim, hparams.prenet_dim],
        )
        self.attention_rnn: nn.LSTMCell = nn.LSTMCell(
            hparams.prenet_dim + self.encoder_embedding_dim,
            hparams.attention_rnn_dim,
        )
        self.attention_layer: Attention = Attention(
            hparams.attention_rnn_dim,
            self.encoder_embedding_dim,
            hparams.attention_dim,
            hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size,
        )
        self.decoder_rnn: nn.LSTMCell = nn.LSTMCell(
            hparams.attention_rnn_dim + self.encoder_embedding_dim,
            hparams.decoder_rnn_dim,
            1,
        )
        self.linear_projection: LinearNorm = LinearNorm(
            hparams.decoder_rnn_dim + self.encoder_embedding_dim,
            hparams.n_mel_channels * hparams.n_frames_per_step,
        )
        self.gate_layer: LinearNorm = LinearNorm(
            hparams.decoder_rnn_dim + self.encoder_embedding_dim,
            1,
            bias=True,
            w_init_gain="sigmoid",
        )

        # Decoder states
        self.attention_hidden: Tensor
        self.attention_cell: Tensor
        self.decoder_hidden: Tensor
        self.decoder_cell: Tensor
        self.attention_weights: Tensor
        self.attention_weights_cum: Tensor
        self.attention_context: Tensor
        self.memory: Tensor
        self.processed_memory: Tensor
        self.mask: Optional[Tensor]

    def get_go_frame(self, memory: Tensor) -> Tensor:
        """
        Args:
            memory (Tensor): (B, T_text, encoder_dim)
        
        Returns:
            Tensor: (B, n_mels * n_frames_per_step) zeros.
        """
        batch_size: int = memory.size(0)
        return memory.new_zeros(batch_size, self.n_mel_channels * self.n_frames_per_step)

    def initialize_decoder_states(self, memory: Tensor, mask: Optional[Tensor]) -> None:
        """
        Args:
            memory (Tensor): (B, T_text, encoder_dim)
            mask (Tensor): (B, T_text)
        """
        batch_size: int = memory.size(0)
        max_time: int = memory.size(1)

        self.attention_hidden = memory.new_zeros(batch_size, self.attention_rnn_dim)
        self.attention_cell = memory.new_zeros(batch_size, self.attention_rnn_dim)
        self.decoder_hidden = memory.new_zeros(batch_size, self.decoder_rnn_dim)
        self.decoder_cell = memory.new_zeros(batch_size, self.decoder_rnn_dim)
        self.attention_weights = memory.new_zeros(batch_size, max_time)
        self.attention_weights_cum = memory.new_zeros(batch_size, max_time)
        self.attention_context = memory.new_zeros(batch_size, self.encoder_embedding_dim)

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory) # (B, T_text, attention_dim)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs: Tensor) -> Tensor:
        """
        Args:
            decoder_inputs (Tensor): (B, n_mels, T_mel)
        
        Returns:
            Tensor: (T_mel_steps, B, n_mels * n_frames_per_step)
        """
        decoder_inputs = decoder_inputs.transpose(1, 2) # (B, T_mel, n_mels)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1) / self.n_frames_per_step),
            -1,
        ) # (B, T_mel_steps, n_mels * n_frames_per_step)
        return decoder_inputs.transpose(0, 1) # (T_mel_steps, B, n_mels * n_frames_per_step)

    def parse_decoder_outputs(self, mel_outputs: List[Tensor], gate_outputs: List[Tensor], alignments: List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            mel_outputs (List[Tensor]): list of (B, n_mels * n_frames_per_step)
            gate_outputs (List[Tensor]): list of (B, 1)
            alignments (List[Tensor]): list of (B, T_text)
        
        Returns:
            Tuple[Tensor, Tensor, Tensor]: mel (B, n_mels, T_mel), gate (B, T_mel), alignments (B, T_mel, T_text).
        """
        alignments_t: Tensor = torch.stack(alignments).transpose(0, 1) # (B, T_mel, T_text)
        
        gate_outputs_t: Tensor = torch.stack(gate_outputs) # (T_mel, B, 1)
        if len(gate_outputs_t.size()) == 1:
            gate_outputs_t = gate_outputs_t.unsqueeze(1)
        gate_outputs_t = gate_outputs_t.transpose(0, 1).contiguous() # (B, T_mel, 1)
        
        mel_outputs_t: Tensor = torch.stack(mel_outputs).transpose(0, 1).contiguous() # (B, T_mel, n_mels * n_frames)
        mel_outputs_t = mel_outputs_t.view(mel_outputs_t.size(0), -1, self.n_mel_channels) # (B, T_mel, n_mels)
        mel_outputs_t = mel_outputs_t.transpose(1, 2) # (B, n_mels, T_mel)
        
        return mel_outputs_t, gate_outputs_t.squeeze(-1), alignments_t

    def decode(self, decoder_input: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            decoder_input (Tensor): (B, n_mels * n_frames_per_step)
        
        Returns:
            Tuple[Tensor, Tensor, Tensor]: mel_output, gate_prediction, attention_weights.
        """
        cell_input: Tensor = torch.cat((decoder_input, self.attention_context), -1) # (B, prenet_dim + encoder_dim)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell)
        ) # (B, attention_rnn_dim)
        
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training
        )
        self.attention_cell = F.dropout(
            self.attention_cell, self.p_attention_dropout, self.training
        )

        attention_weights_cat: Tensor = torch.cat(
            (self.attention_weights.unsqueeze(1), self.attention_weights_cum.unsqueeze(1)),
            dim=1,
        ) # (B, 2, T_text)
        
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden,
            self.memory,
            self.processed_memory,
            attention_weights_cat,
            self.mask,
        ) # (B, encoder_dim), (B, T_text)
        
        self.attention_weights_cum += self.attention_weights

        decoder_rnn_input: Tensor = torch.cat((self.attention_hidden, self.attention_context), -1) # (B, attention_rnn_dim + encoder_dim)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_rnn_input, (self.decoder_hidden, self.decoder_cell)
        ) # (B, decoder_rnn_dim)
        
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training
        )
        self.decoder_cell = F.dropout(
            self.decoder_cell, self.p_decoder_dropout, self.training
        )

        decoder_hidden_attention_context: Tensor = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1
        ) # (B, decoder_rnn_dim + encoder_dim)
        
        decoder_output: Tensor = self.linear_projection(decoder_hidden_attention_context) # (B, n_mels * n_frames)
        gate_prediction: Tensor = self.gate_layer(decoder_hidden_attention_context) # (B, 1)
        
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory: Tensor, decoder_inputs: Tensor, memory_lengths: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            memory (Tensor): (B, T_text, encoder_dim)
            decoder_inputs (Tensor): (B, n_mels, T_mel)
            memory_lengths (Tensor): (B,)
        
        Returns:
            Tuple[Tensor, Tensor, Tensor]: mel_outputs, gate_outputs, alignments.
        """
        decoder_input: Tensor = self.get_go_frame(memory).unsqueeze(0) # (1, B, n_mels * n_frames)
        decoder_inputs_parsed: Tensor = self.parse_decoder_inputs(decoder_inputs) # (T_dec, B, n_mels * n_frames)
        decoder_inputs_parsed = torch.cat((decoder_input, decoder_inputs_parsed), dim=0) # (T_dec+1, B, n_mels * n_frames)
        decoder_inputs_parsed = self.prenet(decoder_inputs_parsed) # (T_dec+1, B, prenet_dim)

        # Frame dropout: randomly zero out entire decoder input frames during training
        # This weakens the autoregressive decoder and forces it to rely on encoder (and z) information
        if self.training and self.p_decoder_input_dropout > 0:
            frame_mask = torch.bernoulli(
                torch.ones(decoder_inputs_parsed.size(0), decoder_inputs_parsed.size(1), 1,
                           device=decoder_inputs_parsed.device) * (1 - self.p_decoder_input_dropout)
            )
            decoder_inputs_parsed = decoder_inputs_parsed * frame_mask

        self.initialize_decoder_states(memory, mask=~get_mask_from_lengths(memory_lengths))

        mel_outputs: List[Tensor] = []
        gate_outputs: List[Tensor] = []
        alignments: List[Tensor] = []
        
        while len(mel_outputs) < decoder_inputs_parsed.size(0) - 1:
            # Scheduled sampling: sometimes feed model's own prediction instead of ground truth
            if self.training and self.p_scheduled_sampling > 0.0 and len(mel_outputs) > 0:
                if torch.rand(1, device=decoder_inputs_parsed.device).item() < self.p_scheduled_sampling:
                    # Use model's previous prediction (no frame dropout, same as inference)
                    step_input = self.prenet(mel_outputs[-1].detach())
                else:
                    step_input = decoder_inputs_parsed[len(mel_outputs)]
            else:
                step_input = decoder_inputs_parsed[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(step_input)
            mel_outputs.append(mel_output)
            gate_outputs.append(gate_output)
            alignments.append(attention_weights)

        return self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments)

    def inference(self, memory: Tensor, force_monotonic: bool = False, monotonic_window: int = 3,
                  attn_stop_frames: int = 0) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            memory (Tensor): (B, T_text, encoder_dim)
            force_monotonic (bool): If True, prevent attention from moving backward.
            monotonic_window (int): Number of positions backward the attention is allowed to revisit.
            attn_stop_frames (int): Stop when attention peak stays on the last encoder position for
                this many consecutive frames (0 = disabled). Reliably stops without a trained gate.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: mel_outputs, gate_outputs, alignments.
        """
        decoder_input: Tensor = self.get_go_frame(memory) # (B, n_mels * n_frames)
        self.initialize_decoder_states(memory, mask=None)

        mel_outputs: List[Tensor] = []
        gate_outputs: List[Tensor] = []
        alignments: List[Tensor] = []
        monotonic_peak: int = 0  # tracks furthest attended encoder position
        last_enc_pos: int = memory.shape[1] - 1  # index of final encoder token
        attn_end_count: int = 0  # consecutive frames with peak at last encoder position

        while True:
            step_input: Tensor = self.prenet(decoder_input) # (B, prenet_dim)
            mel_output, gate_output, attention_weights = self.decode(step_input)
            mel_outputs.append(mel_output)
            gate_outputs.append(gate_output)
            alignments.append(attention_weights)

            current_peak = int(attention_weights.argmax(dim=-1).max().item())

            if force_monotonic:
                if current_peak > monotonic_peak:
                    monotonic_peak = current_peak
                cutoff = max(0, monotonic_peak - monotonic_window)
                if cutoff > 0:
                    mono_mask = torch.zeros(memory.shape[0], memory.shape[1],
                                           dtype=torch.bool, device=memory.device)
                    mono_mask[:, :cutoff] = True
                    self.mask = mono_mask

            # Attention-peak-based stopping: gate alternative for exposure-bias-affected models
            if attn_stop_frames > 0:
                if current_peak >= last_enc_pos:
                    attn_end_count += 1
                else:
                    attn_end_count = 0
                if attn_end_count >= attn_stop_frames:
                    break

            if torch.sigmoid(gate_output).max() > self.gate_threshold:
                break
            if len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break
            decoder_input = mel_output # Autoregressive feedback

        return self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments)


class Tacotron2(nn.Module):
    """
    Tacotron 2 Variational Autoencoder.

    Architecture:
        TextEmbedding -> Encoder -> VAE_GST -> Decoder -> Postnet

    Inputs:
        inputs:
            text_padded: (B, T_text)
            input_lengths: (B,)
            mel_padded: (B, n_mels, T_mel)
            output_lengths: (B,)

    Outputs:
        mel_outputs: (B, n_mels, T_mel)
        mel_outputs_postnet: (B, n_mels, T_mel)
        gate_outputs: (B, T_mel)
        alignments: (B, T_mel, T_text)
        mu: (B, L)
        logvar: (B, L)
        z: (B, L)

    Example:
        >>> model = Tacotron2(hparams)
        >>> outputs = model(batch)
    """
    def __init__(self, hparams: Tacotron2VAEHparams) -> None:
        """
        Initialize Tacotron 2.

        Args:
            hparams (Tacotron2VAEHparams): Model hyperparameters.
        """
        super().__init__()
        self.hparams: Tacotron2VAEHparams = hparams
        self.mask_padding: bool = hparams.mask_padding
        self.n_mel_channels: int = hparams.n_mel_channels
        self.n_frames_per_step: int = hparams.n_frames_per_step

        self.transcript_embedding: nn.Embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim
        )

        std: float = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val: float = sqrt(3.0) * std
        self.transcript_embedding.weight.data.uniform_(-val, val)

        self.encoder: Encoder = Encoder(hparams)
        self.decoder: Decoder = Decoder(hparams)
        self.postnet: Postnet = Postnet(hparams)
        self.vae_gst: VAE_GST = VAE_GST(hparams)

    def parse_batch(self, batch: Any, device: torch.device) -> Tuple[Tuple[Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]]:
        """
        Args:
            batch: Batch from DataLoader.
            device: Target device.
        
        Returns:
            inputs: (text, input_lengths, mel, output_lengths)
            targets: (mel, gate)
        """
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

    def parse_output(self, outputs: List[Tensor], output_lengths: Optional[Tensor] = None) -> List[Tensor]:
        """
        Args:
            outputs: List of model outputs.
            output_lengths: (B,)
        
        Returns:
            outputs: Masked model outputs.
        """
        if self.mask_padding and output_lengths is not None:
            mask: Tensor = ~get_mask_from_lengths(output_lengths) # (B, T_mel)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1)) # (n_mels, B, T_mel)
            mask = mask.permute(1, 0, 2) # (B, n_mels, T_mel)
            
            outputs[0] = outputs[0].masked_fill(mask, -11.5129) # mel_outputs
            outputs[1] = outputs[1].masked_fill(mask, -11.5129) # mel_outputs_postnet
            outputs[2] = outputs[2].masked_fill(mask[:, 0, :], 1e3) # gate_outputs (padded positions set to high logic)
        return outputs

    def forward(self, inputs: Tuple[Tensor, Tensor, Tensor, Tensor]) -> List[Tensor]:
        """
        Forward pass of Tacotron 2.

        Args:
            inputs (Tuple[Tensor, Tensor, Tensor, Tensor]): 
                (text_padded, input_lengths, mel_padded, output_lengths)

        Returns:
            List[Tensor]: [mel, mel_postnet, gate, alignments, mu, logvar, z]
        """
        text_padded, input_lengths, targets, output_lengths = inputs

        # Text encoding
        embedded_inputs: Tensor = self.transcript_embedding(text_padded).transpose(1, 2) # (B, embed_dim, T_text)
        transcript_outputs: Tensor = self.encoder(embedded_inputs, input_lengths) # (B, T_text, encoder_dim)

        # Prosody encoding (VAE-GST)
        prosody_outputs: Tensor
        mu: Tensor
        logvar: Tensor
        z: Tensor
        prosody_outputs, mu, logvar, z = self.vae_gst(targets) # (B, E), (B, L), (B, L), (B, L)
        
        # Inject prosody into encoder outputs
        prosody_outputs = prosody_outputs.unsqueeze(1).expand_as(transcript_outputs) # (B, T_text, encoder_dim)
        encoder_outputs: Tensor = transcript_outputs + prosody_outputs # (B, T_text, encoder_dim)

        # Autoregressive decoding
        mel_outputs: Tensor
        gate_outputs: Tensor
        alignments: Tensor
        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, targets, memory_lengths=input_lengths
        ) # (B, n_mels, T_mel), (B, T_mel), (B, T_mel, T_text)

        # Postnet refinement
        mel_outputs_postnet: Tensor = self.postnet(mel_outputs) # (B, n_mels, T_mel)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet # (B, n_mels, T_mel)

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments, mu, logvar, z],
            output_lengths,
        )

    @torch.inference_mode()
    def infer(self, text: Tensor, audio: Tensor, waveglow: nn.Module) -> Tensor:
        """
        Generate audio from text conditioned on reference audio using WaveGlow.

        Args:
            text (Tensor): Text sequence tensor (1, T_text).
            audio (Tensor): Reference mel-spectrogram tensor (1, n_mels, T_mel).
            waveglow (nn.Module): Pre-trained WaveGlow model for waveform synthesis.

        Returns:
            Tensor: Synthesized audio waveform.
        """
        # Encode text
        embedded_inputs = self.transcript_embedding(text).transpose(1, 2)
        transcript_outputs = self.encoder.inference(embedded_inputs)

        # Extract prosody from reference audio
        latent_vector, _, _, _ = self.vae_gst(audio)
        
        # Inject prosody
        latent_vector = latent_vector.unsqueeze(1).expand_as(transcript_outputs)
        encoder_outputs = transcript_outputs + latent_vector
        
        # Decode to mel-spectrogram
        mel_outputs, _, _ = self.decoder.inference(encoder_outputs)
        
        # Refine mel-spectrogram via postnet
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        
        # Generate audio via WaveGlow
        audio_output = waveglow.infer(mel_outputs_postnet, sigma=0.6)
        
        return audio_output

    @torch.inference_mode()
    def inference_mel(self, text: Tensor, audio: Tensor,
                      force_monotonic: bool = False, monotonic_window: int = 3,
                      attn_stop_frames: int = 0) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Generate mel-spectrogram from text conditioned on reference audio.

        Args:
            text (Tensor): Text sequence tensor (1, T_text).
            audio (Tensor): Reference mel-spectrogram tensor (1, n_mels, T_mel).
            force_monotonic (bool): Prevent attention from attending backward (fixes attention looping).
            monotonic_window (int): How many encoder positions backward the attention may still revisit.
            attn_stop_frames (int): Stop when attention peak stays on last encoder pos for N frames (0=off).

        Returns:
            Tuple[Tensor, Tensor, Tensor]: mel_outputs, mel_outputs_postnet, alignments
        """
        # Encode text
        embedded_inputs = self.transcript_embedding(text).transpose(1, 2)
        transcript_outputs = self.encoder.inference(embedded_inputs)

        # Extract prosody from reference audio
        latent_vector, _, _, _ = self.vae_gst(audio)

        # Inject prosody
        latent_vector = latent_vector.unsqueeze(1).expand_as(transcript_outputs)
        encoder_outputs = transcript_outputs + latent_vector

        # Decode to mel-spectrogram
        mel_outputs, _, alignments = self.decoder.inference(
            encoder_outputs, force_monotonic=force_monotonic, monotonic_window=monotonic_window,
            attn_stop_frames=attn_stop_frames,
        )

        # Refine mel-spectrogram via postnet
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return mel_outputs, mel_outputs_postnet, alignments



def load_tacotron2_vae_model(
    hparams: Tacotron2VAEHparams,
    device: Optional[torch.device] = None,
) -> Tacotron2:
    """
    Load a Tacotron2 VAE model.

    Args:
        hparams (Tacotron2VAEHparams): Model hyperparameters.
        device (Optional[torch.device]): Target device.

    Returns:
        Tacotron2: Initialized model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: Tacotron2 = Tacotron2(hparams).to(device)
    return model


def get_model_size_info(model: Tacotron2) -> Dict[str, int]:
    """
    Get model parameter counts.

    Args:
        model (Tacotron2): The model.

    Returns:
        Dict[str, int]: total_params, trainable_params.
    """
    total: int = sum(p.numel() for p in model.parameters())
    trainable: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


def load_pretrained_tacotron2_backbone(
    model: Tacotron2,
    pretrained_path: str,
    device: Optional[torch.device] = None,
) -> Tacotron2:
    """
    Load pretrained NVIDIA Tacotron2 weights into the Tacotron2-VAE model.

    Maps the NVIDIA checkpoint key format (module.X) to the VAE model's key format.
    Only loads weights for the backbone components (encoder, decoder, postnet, embedding).
    VAE-specific components (vae_gst) are left randomly initialized.

    If n_symbols differs between pretrained and current model, the embedding weights
    are partially loaded (only the overlapping rows).

    Args:
        model (Tacotron2): The Tacotron2-VAE model to load weights into.
        pretrained_path (str): Path to the NVIDIA Tacotron2 checkpoint file.
        device (Optional[torch.device]): Target device.

    Returns:
        Tacotron2: Model with pretrained backbone weights loaded.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint: Dict = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    pretrained_sd: Dict[str, Tensor] = checkpoint.get("state_dict", checkpoint)

    # Build mapping from NVIDIA keys to VAE model keys
    # NVIDIA format: "module.encoder.xxx" -> VAE format: "encoder.xxx"
    # NVIDIA format: "module.embedding.weight" -> VAE format: "transcript_embedding.weight"
    key_mapping: Dict[str, str] = {}
    for nvidia_key in pretrained_sd.keys():
        # Strip "module." prefix (from DataParallel wrapping)
        if nvidia_key.startswith("module."):
            clean_key = nvidia_key[len("module."):]
        else:
            clean_key = nvidia_key

        # Map "embedding" -> "transcript_embedding"
        if clean_key.startswith("embedding."):
            clean_key = "transcript_" + clean_key

        key_mapping[nvidia_key] = clean_key

    model_sd = model.state_dict()
    loaded_keys: list = []
    skipped_keys: list = []

    for nvidia_key, vae_key in key_mapping.items():
        if vae_key not in model_sd:
            skipped_keys.append(f"{nvidia_key} -> {vae_key} (not in model)")
            continue

        pretrained_tensor = pretrained_sd[nvidia_key]
        model_tensor = model_sd[vae_key]

        # Handle shape mismatch for embedding (different n_symbols)
        if pretrained_tensor.shape != model_tensor.shape:
            if "transcript_embedding" in vae_key:
                # Load only overlapping rows
                n_overlap = min(pretrained_tensor.shape[0], model_tensor.shape[0])
                model_sd[vae_key][:n_overlap] = pretrained_tensor[:n_overlap]
                
                # Initialize new rows using the statistics of the pretrained embeddings
                if model_tensor.shape[0] > n_overlap:
                    pt_mean = pretrained_tensor.mean()
                    pt_std = pretrained_tensor.std()
                    # Apply N(mean, std) to the newly added phonemes
                    torch.nn.init.normal_(model_sd[vae_key][n_overlap:], mean=pt_mean.item(), std=pt_std.item())
                    
                loaded_keys.append(f"{vae_key} (partial: {n_overlap}/{model_tensor.shape[0]} rows, new initialized with std={pt_std.item():.4f})")
            else:
                skipped_keys.append(
                    f"{nvidia_key} -> {vae_key} (shape mismatch: "
                    f"{pretrained_tensor.shape} vs {model_tensor.shape})"
                )
            continue

        model_sd[vae_key] = pretrained_tensor
        loaded_keys.append(vae_key)

    model.load_state_dict(model_sd)
    model = model.to(device)

    print(f"[Pretrained Backbone] Loaded {len(loaded_keys)} weight tensors")
    print(f"[Pretrained Backbone] Skipped {len(skipped_keys)} weight tensors")
    if skipped_keys:
        for sk in skipped_keys[:10]:
            print(f"  Skipped: {sk}")

    # VAE-specific components that were NOT loaded (kept random init):
    vae_keys = [k for k in model_sd.keys() if "vae_gst" in k]
    print(f"[Pretrained Backbone] VAE components ({len(vae_keys)} tensors) kept with random init")

    return model
