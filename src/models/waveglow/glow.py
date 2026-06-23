"""
WaveGlow implementation.

Responsibilities:
    - Implement WaveGlow: A flow-based generative model for audio synthesis.
    - Implement WN: Non-causal WaveNet-like architecture for affine coupling.
    - Implement Invertible1x1Conv: Reversible 1x1 convolution for flow mixing.
    - Implement WaveGlowLoss: Negative Log-Likelihood (NLL) loss for the Gaussian prior.

Main Classes:
    - WaveGlow: Top-level model.
    - WN: Core transform module.
    - Invertible1x1Conv: Mixer layer.
    - WaveGlowLoss: Loss function.

Main Functions:
    - fused_add_tanh_sigmoid_multiply: JIT-compiled activation function.
    - remove: Helper to strip weight normalization.

Tensor Conventions:
    B = batch size
    T = sequence length (audio samples or frames)
    n_mels = mel frequency bins
    n_group = grouping size for squeeze operation
"""
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple, List, Union, Dict, Any


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a: Tensor, input_b: Tensor, n_channels: Tensor) -> Tensor:
    """
    Fuses the add, tanh, sigmoid, and multiply operations for performance.

    Architecture:
        (a + b) -> Split -> [Tanh(left) * Sigmoid(right)]

    Args:
        input_a (Tensor): First input. Shape: (B, 2*C, T)
        input_b (Tensor): Second input. Shape: (B, 2*C, T)
        n_channels (Tensor): Scalar tensor with value C.

    Returns:
        Tensor: Processed output. Shape: (B, C, T)
    """
    n_channels_int: int = int(n_channels[0].item())
    in_act: Tensor = input_a + input_b  # (B, 2*C, T)
    t_act: Tensor = torch.tanh(in_act[:, :n_channels_int, :])  # (B, C, T)
    s_act: Tensor = torch.sigmoid(in_act[:, n_channels_int:, :])  # (B, C, T)
    acts: Tensor = t_act * s_act  # (B, C, T)
    return acts


class WaveGlowLoss(torch.nn.Module):
    """
    Negative Log-Likelihood Loss for WaveGlow.

    Mathematical Intuition:
        Loss = (z^2 / 2*sigma^2) - sum(log|s|) - sum(log|det W|)

    Inputs:
        model_output:
            z: Latent variable. (B, C_latent, T_groups)
            log_s_list: List of log-scale tensors from affine coupling.
            log_det_W_list: List of log-determinant scalars from 1x1 convs.

    Outputs:
        loss: Scalar NLL.
    """
    def __init__(self, sigma: float = 1.0) -> None:
        """
        Initialize the loss.

        Args:
            sigma (float): Standard deviation of Gaussian prior.
        """
        super(WaveGlowLoss, self).__init__()
        self.sigma: float = sigma

    def forward(self, model_output: Tuple[Tensor, List[Tensor], List[Tensor]]) -> Tensor:
        """
        Compute WaveGlow NLL loss.

        Args:
            model_output: (z, log_s_list, log_det_W_list).

        Returns:
            Tensor: Scalar loss normalized by dimensions.
        """
        z, log_s_list, log_det_W_list = model_output
        log_s_total: Tensor
        log_det_W_total: Tensor

        for i, log_s in enumerate(log_s_list):
            if i == 0:
                log_s_total = torch.sum(log_s)  # scalar
                log_det_W_total = log_det_W_list[i]  # scalar
            else:
                log_s_total = log_s_total + torch.sum(log_s)
                log_det_W_total = log_det_W_total + log_det_W_list[i]

        # Prior term + Change of variables correction
        loss: Tensor = torch.sum(z*z)/(2*self.sigma*self.sigma) - log_s_total - log_det_W_total # scalar
        
        # Normalize by batch and all time/channel steps
        return loss/(z.size(0)*z.size(1)*z.size(2))


class Invertible1x1Conv(torch.nn.Module):
    """
    Invertible 1x1 Convolution for Normalizing Flows.

    Architecture:
        1x1 Conv with weight initialized as orthonormal matrix.
    
    Inputs:
        z:
            Shape (B, C, T_groups)
        reverse:
            Whether to apply the inverse operation.

    Outputs:
        output:
            Shape (B, C, T_groups)
        log_det (if forward):
            Scalar.
    """
    def __init__(self, c: int) -> None:
        """
        Initialize the invertible layer.

        Args:
            c (int): Number of channels.
        """
        super(Invertible1x1Conv, self).__init__()
        self.conv: torch.nn.Conv1d = torch.nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0, bias=False)

        # Sample a random orthonormal matrix
        W: Tensor = torch.linalg.qr(torch.FloatTensor(c, c).normal_())[0] # (C, C)

        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:,0] = -1*W[:,0]
        
        W = W.view(c, c, 1)
        self.conv.weight.data = W

    def forward(self, z: Tensor, reverse: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Args:
            z (Tensor): Input (B, C, T_groups).
            reverse (bool): Inverse flag.

        Returns:
            Forward: (output, log_det).
            Reverse: output.
        """
        batch_size, group_size, n_of_groups = z.size()

        W: Tensor = self.conv.weight.squeeze()  # (C, C)

        if reverse:
            # Recompute W_inverse if not cached
            if not hasattr(self, 'W_inverse') or self.W_inverse is None:
                W_inverse: Tensor = W.float().inverse()  # (C, C)
                W_inverse = W_inverse[..., None]  # (C, C, 1)
                W_inverse = W_inverse.to(device=z.device, dtype=z.dtype)
                self.W_inverse: Tensor = W_inverse
            
            z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0) # (B, C, T_groups)
            return z
        else:
            # Use slogdet[1] to avoid NaN if determinant is negative
            _, log_det_W_val = torch.slogdet(W)
            log_det_W: Tensor = batch_size * n_of_groups * log_det_W_val # scalar
            z = self.conv(z) # (B, C, T_groups)
            
            # Invalidate cache if forward pass is called (which means training occurred)
            if hasattr(self, 'W_inverse'):
                self.W_inverse = None
                
            return z, log_det_W


class WN(torch.nn.Module):
    """
    Non-causal WaveNet-like transformation block.

    Architecture:
        Conv1d (Start) -> [Dilation Conv + Tanh/Sigmoid Cond] * n_layers -> Conv1d (End)

    Inputs:
        forward_input:
            (audio, spect)
            audio: (B, C_in, T_groups)
            spect: (B, C_mel_group, T_groups)

    Outputs:
        output:
            Shape (B, 2*C_in, T_groups) (scale and bias for affine coupling).
    """
    def __init__(self, n_in_channels: int, n_mel_channels: int, n_layers: int, n_channels: int, kernel_size: int) -> None:
        """
        Initialize the WN block.

        Args:
            n_in_channels (int): Input audio channels.
            n_mel_channels (int): Spectrogram channels (pre-upsampled).
            n_layers (int): Layers per block.
            n_channels (int): Residual/skip dimension.
            kernel_size (int): Kernel size.
        """
        super(WN, self).__init__()
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        self.n_layers: int = n_layers
        self.n_channels: int = n_channels
        self.in_layers: torch.nn.ModuleList = torch.nn.ModuleList()
        self.res_skip_layers: torch.nn.ModuleList = torch.nn.ModuleList()
        self.cond_layers: torch.nn.ModuleList = torch.nn.ModuleList()

        self.start: torch.nn.Conv1d = torch.nn.utils.weight_norm(
            torch.nn.Conv1d(n_in_channels, n_channels, 1), name='weight'
        )

        # Output produces parameters for affine transform (log_s and b)
        self.end: torch.nn.Conv1d = torch.nn.Conv1d(n_channels, 2*n_in_channels, 1)
        self.end.weight.data.zero_()
        self.end.bias.data.zero_()

        for i in range(n_layers):
            dilation: int = 2 ** i
            padding: int = int((kernel_size*dilation - dilation)/2)
            
            in_layer: torch.nn.Conv1d = torch.nn.utils.weight_norm(
                torch.nn.Conv1d(n_channels, 2*n_channels, kernel_size, dilation=dilation, padding=padding),
                name='weight'
            )
            self.in_layers.append(in_layer)

            cond_layer: torch.nn.Conv1d = torch.nn.utils.weight_norm(
                torch.nn.Conv1d(n_mel_channels, 2*n_channels, 1),
                name='weight'
            )
            self.cond_layers.append(cond_layer)

            res_skip_channels: int = 2*n_channels if i < n_layers - 1 else n_channels
            res_skip_layer: torch.nn.Conv1d = torch.nn.utils.weight_norm(
                torch.nn.Conv1d(n_channels, res_skip_channels, 1),
                name='weight'
            )
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, forward_input: Tuple[Tensor, Tensor]) -> Tensor:
        """
        Forward computation.

        Args:
            forward_input: (audio, spect).

        Returns:
            Tensor: (B, 2*C_in, T_groups)
        """
        audio, spect = forward_input
        audio = self.start(audio)  # (B, n_channels, T_groups)

        output: Tensor = torch.zeros_like(audio) # (B, n_channels, T_groups)

        for i in range(self.n_layers):
            acts: Tensor = fused_add_tanh_sigmoid_multiply(
                self.in_layers[i](audio),  # (B, 2*n_channels, T_groups)
                self.cond_layers[i](spect),  # (B, 2*n_channels, T_groups)
                torch.IntTensor([self.n_channels]))  # (B, n_channels, T_groups)

            res_skip_acts: Tensor = self.res_skip_layers[i](acts)  # (B, res_skip_channels, T_groups)
            
            if i < self.n_layers - 1:
                audio = res_skip_acts[:, :self.n_channels, :] + audio  # Residual path
                skip_acts = res_skip_acts[:, self.n_channels:, :]  # Skip path
            else:
                skip_acts = res_skip_acts

            if i == 0:
                output = skip_acts
            else:
                output = skip_acts + output
        
        return self.end(output)  # (B, 2*n_in_channels, T_groups)


class WaveGlow(torch.nn.Module):
    """
    WaveGlow: A Flow-based Generative Model for Voice Synthesis.

    Architecture:
        Upsample -> [Squeeze -> 1x1Conv -> AffineCoupling (WN) -> Split] * n_flows

    Inputs (Training):
        forward_input: (spect, audio)
            spect: (B, n_mel_channels, frames)
            audio: (B, time)

    Outputs (Training):
        z: (B, C_latent, T_groups)
        log_s_list: list of log scales.
        log_det_W_list: list of log determinants.

    Inputs (Inference):
        spect: (B, n_mel_channels, frames)
        sigma: scalar noise variance.

    Outputs (Inference):
        audio: (B, time)

    Example:
        >>> waveglow = WaveGlow(...)
        >>> audio = waveglow.infer(mel_spectrogram)
    """
    def __init__(
        self, 
        n_mel_channels: int, 
        n_flows: int, 
        n_group: int, 
        n_early_every: int,
        n_early_size: int, 
        WN_config: Dict[str, Any]
    ) -> None:
        """
        Initialize WaveGlow.

        Args:
            n_mel_channels (int): Mel bins.
            n_flows (int): Flow steps.
            n_group (int): Number of audio samples to group.
            n_early_every (int): Frequency of early sample output.
            n_early_size (int): Dimensions of early output.
            WN_config (dict): WaveNet configuration.
        """
        super(WaveGlow, self).__init__()

        self.upsample: torch.nn.ConvTranspose1d = torch.nn.ConvTranspose1d(
            n_mel_channels, n_mel_channels, 1024, stride=256
        )
        assert(n_group % 2 == 0)
        self.n_flows: int = n_flows
        self.n_group: int = n_group
        self.n_early_every: int = n_early_every
        self.n_early_size: int = n_early_size
        self.WN: torch.nn.ModuleList = torch.nn.ModuleList()
        self.convinv: torch.nn.ModuleList = torch.nn.ModuleList()

        n_half: int = int(n_group/2)
        n_remaining_channels: int = n_group

        for k in range(n_flows):
            if k % self.n_early_every == 0 and k > 0:
                n_half = n_half - int(self.n_early_size/2)
                n_remaining_channels = n_remaining_channels - self.n_early_size
            
            self.convinv.append(Invertible1x1Conv(n_remaining_channels))
            self.WN.append(WN(n_half, n_mel_channels*n_group, **WN_config))
        
        self.n_remaining_channels: int = n_remaining_channels

    def forward(self, forward_input: Tuple[Tensor, Tensor]) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        """
        Forward pass (training). Transforms audio to Gaussian latent z.

        Args:
            forward_input: (spect, audio).
                spect: (B, C_mel, T_f)
                audio: (B, T_a)

        Returns:
            (z, log_s_list, log_det_W_list).
        """
        spect, audio = forward_input

        # 1. Upsample spectrogram to match audio resolution
        spect = self.upsample(spect)  # (B, C_mel, T_a_upsampled)
        assert(spect.size(2) >= audio.size(1))
        if spect.size(2) > audio.size(1):
            spect = spect[:, :, :audio.size(1)]  # (B, C_mel, T_a)

        # 2. Reshape spectrogram to group level
        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3) # (B, T_groups, C_mel, group_size)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1) # (B, C_mel*n_group, T_groups)

        # 3. Reshape audio to group level
        audio = audio.unfold(1, self.n_group, self.n_group).permute(0, 2, 1) # (B, n_group, T_groups)
        
        output_audio: List[Tensor] = []
        log_s_list: List[Tensor] = []
        log_det_W_list: List[Tensor] = []

        for k in range(self.n_flows):
            if k % self.n_early_every == 0 and k > 0:
                output_audio.append(audio[:, :self.n_early_size, :]) # Save early output
                audio = audio[:, self.n_early_size:, :] # Continue with remaining dimensions

            # Mixer
            audio, log_det_W = self.convinv[k](audio) # (B, C_rem, T_groups)
            log_det_W_list.append(log_det_W)

            # Affine Coupling
            n_half: int = int(audio.size(1)/2)
            audio_0: Tensor = audio[:, :n_half, :]
            audio_1: Tensor = audio[:, n_half:, :]

            output: Tensor = self.WN[k]((audio_0, spect)) # (B, 2*n_half, T_groups)
            log_s: Tensor = output[:, n_half:, :]
            b: Tensor = output[:, :n_half, :]
            
            audio_1 = torch.exp(log_s)*audio_1 + b
            log_s_list.append(log_s)

            audio = torch.cat([audio_0, audio_1], 1)

        output_audio.append(audio)
        return torch.cat(output_audio, 1), log_s_list, log_det_W_list

    def infer(self, spect: Tensor, sigma: float = 1.0) -> Tensor:
        """
        Reverse pass (inference). Transforms Gaussian noise to audio.

        Args:
            spect (Tensor): Mel-spectrogram (B, C_mel, T_f).
            sigma (float): Sampling variance.

        Returns:
            Tensor: Synthesized audio (B, T_a).
        """
        spect = self.upsample(spect)  # (B, C_mel, T_a_ups)
        
        # Trim artifacts
        time_cutoff: int = self.upsample.kernel_size[0] - self.upsample.stride[0]
        spect = spect[:, :, :-time_cutoff] # (B, C_mel, T_a)

        # Group spectrogram
        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1) # (B, C_mel*group, T_groups)

        # Initialize with Gaussian noise
        audio: Tensor = torch.randn(
            spect.size(0), self.n_remaining_channels, spect.size(2),
            dtype=spect.dtype, device=spect.device
        )
        audio = sigma * audio  # (B, C_rem, T_groups)

        for k in reversed(range(self.n_flows)):
            n_half: int = int(audio.size(1)/2)
            audio_0: Tensor = audio[:, :n_half, :]
            audio_1: Tensor = audio[:, n_half:, :]

            output: Tensor = self.WN[k]((audio_0, spect))
            s: Tensor = output[:, n_half:, :]
            b: Tensor = output[:, :n_half, :]
            
            # Inverse affine transform
            audio_1 = (audio_1 - b)/torch.exp(s)
            audio = torch.cat([audio_0, audio_1], 1)

            # Inverse mixer
            audio = self.convinv[k](audio, reverse=True)

            # Re-inject early samples from prior
            if k % self.n_early_every == 0 and k > 0:
                z: Tensor = torch.randn(
                    spect.size(0), self.n_early_size, spect.size(2),
                    dtype=spect.dtype, device=spect.device
                )
                audio = torch.cat((sigma*z, audio), 1)

        # Reshape back to 1D waveform
        audio_final: Tensor = audio.permute(0, 2, 1).contiguous().view(audio.size(0), -1) # (B, T_a)
        return audio_final

    @staticmethod
    def remove_weightnorm(model: torch.nn.Module) -> torch.nn.Module:
        """
        Strips weight normalization from all layers.

        Args:
            model (nn.Module): WaveGlow instance.

        Returns:
            nn.Module: Optimized model.
        """
        waveglow = model
        for WN_block in waveglow.WN:
            WN_block.start = torch.nn.utils.remove_weight_norm(WN_block.start)
            WN_block.in_layers = remove(WN_block.in_layers)
            WN_block.cond_layers = remove(WN_block.cond_layers)
            WN_block.res_skip_layers = remove(WN_block.res_skip_layers)
        return waveglow


def remove(conv_list: torch.nn.ModuleList) -> torch.nn.ModuleList:
    """
    Removes weight normalization from a list of layers.

    Args:
        conv_list (ModuleList): Layers.

    Returns:
        ModuleList: Unnormalized layers.
    """
    new_conv_list: torch.nn.ModuleList = torch.nn.ModuleList()
    for old_conv in conv_list:
        new_conv: torch.nn.Module = torch.nn.utils.remove_weight_norm(old_conv)
        new_conv_list.append(new_conv)
    return new_conv_list
