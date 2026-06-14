import copy
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from typing import Tuple, List, Union, Dict, Any


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a: torch.Tensor, input_b: torch.Tensor, n_channels: torch.Tensor) -> torch.Tensor:
    """
    Fuses the add, tanh, sigmoid, and multiply operations for performance.

    This function adds two inputs, splits the result into two halves along the channel dimension,
    applies tanh to the first half and sigmoid to the second half, and then multiplies them.

    Args:
        input_a (torch.Tensor): The first input tensor. Shape: (batch_size, 2*n_channels, length)
        input_b (torch.Tensor): The second input tensor. Shape: (batch_size, 2*n_channels, length)
        n_channels (torch.Tensor): A tensor containing a single integer representing the number of channels (half of the input channels).

    Returns:
        torch.Tensor: The result of the operation. Shape: (batch_size, n_channels, length)

    Example:
        >>> input_a = torch.randn(2, 4, 10)
        >>> input_b = torch.randn(2, 4, 10)
        >>> n_channels = torch.IntTensor([2])
        >>> output = fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels)
        >>> output.shape
        torch.Size([2, 2, 10])
    """
    n_channels_int = n_channels[0]
    in_act = input_a + input_b  # [batch_size, 2*n_channels, length]
    t_act = torch.tanh(in_act[:, :n_channels_int, :])  # [batch_size, n_channels, length]
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])  # [batch_size, n_channels, length]
    acts = t_act * s_act  # [batch_size, n_channels, length]
    return acts


class WaveGlowLoss(torch.nn.Module):
    """
    Loss function for WaveGlow.

    Computes the negative log-likelihood loss for the WaveGlow model.

    Args:
        sigma (float, optional): The standard deviation of the Gaussian prior. Defaults to 1.0.

    Example:
        >>> loss_fn = WaveGlowLoss(sigma=1.0)
        >>> z = torch.randn(2, 8, 100)
        >>> log_s_list = [torch.randn(2, 4, 100) for _ in range(12)]
        >>> log_det_W_list = [torch.tensor(0.5) for _ in range(12)]
        >>> loss = loss_fn((z, log_s_list, log_det_W_list))
        >>> loss.item() # Returns a scalar loss
    """
    def __init__(self, sigma: float = 1.0):
        super(WaveGlowLoss, self).__init__()
        self.sigma = sigma

    def forward(self, model_output: Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]) -> torch.Tensor:
        """
        Calculates the loss.

        Args:
            model_output (Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]): Output from WaveGlow forward pass.

        Returns:
            torch.Tensor: A scalar loss value.
        """
        z, log_s_list, log_det_W_list = model_output
        for i, log_s in enumerate(log_s_list):
            if i == 0:
                log_s_total = torch.sum(log_s)  # []
                log_det_W_total = log_det_W_list[i]  # []
            else:
                log_s_total = log_s_total + torch.sum(log_s)  # []
                log_det_W_total += log_det_W_list[i]  # []

        loss = torch.sum(z*z)/(2*self.sigma*self.sigma) - log_s_total - log_det_W_total  # []
        return loss/(z.size(0)*z.size(1)*z.size(2))  # []


class Invertible1x1Conv(torch.nn.Module):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix. If reverse=True it does convolution with inverse.

    Args:
        c (int): Number of channels.

    Example:
        >>> conv = Invertible1x1Conv(8)
        >>> z = torch.randn(2, 8, 100)
        >>> out, log_det = conv(z)
        >>> z_rev = conv(out, reverse=True)
    """
    def __init__(self, c: int):
        super(Invertible1x1Conv, self).__init__()
        self.conv = torch.nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0, bias=False)

        # Sample a random orthonormal matrix to initialize weights
        W = torch.qr(torch.FloatTensor(c, c).normal_())[0]

        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:,0] = -1*W[:,0]
        W = W.view(c, c, 1)
        self.conv.weight.data = W

    def forward(self, z: torch.Tensor, reverse: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward or reverse computation of the 1x1 convolution.

        Args:
            z (torch.Tensor): Input tensor. Shape: (batch_size, channels, length)
            reverse (bool, optional): Whether to apply the inverse. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: The output tensor, or tuple of output tensor and log determinant.
        """
        # shape
        batch_size, group_size, n_of_groups = z.size()

        W = self.conv.weight.squeeze()  # [c, c]

        if reverse:
            if not hasattr(self, 'W_inverse'):
                # Reverse computation
                W_inverse = W.float().inverse()  # [c, c]
                W_inverse = Variable(W_inverse[..., None])  # [c, c, 1]
                W_inverse = W_inverse.to(device=z.device, dtype=z.dtype)
                self.W_inverse = W_inverse
            z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)  # [batch_size, group_size, n_of_groups]
            return z
        else:
            # Forward computation
            log_det_W = batch_size * n_of_groups * torch.logdet(W)  # []
            z = self.conv(z)  # [batch_size, group_size, n_of_groups]
            return z, log_det_W


class WN(torch.nn.Module):
    """
    This is the WaveNet like layer for the affine coupling. The primary difference
    from WaveNet is the convolutions need not be causal. There is also no dilation
    size reset. The dilation only doubles on each layer.

    Args:
        n_in_channels (int): Input audio channels.
        n_mel_channels (int): Input spectrogram channels.
        n_layers (int): Number of WaveNet layers.
        n_channels (int): Number of residual and skip channels.
        kernel_size (int): Kernel size for dilated convolutions.

    Example:
        >>> wn = WN(n_in_channels=4, n_mel_channels=80, n_layers=8, n_channels=256, kernel_size=3)
        >>> audio = torch.randn(2, 4, 100)
        >>> spect = torch.randn(2, 80, 100)
        >>> output = wn((audio, spect))
        >>> output.shape
        torch.Size([2, 8, 100])
    """
    def __init__(self, n_in_channels: int, n_mel_channels: int, n_layers: int, n_channels: int, kernel_size: int):
        super(WN, self).__init__()
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.cond_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(n_in_channels, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        end = torch.nn.Conv1d(n_channels, 2*n_in_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        for i in range(n_layers):
            dilation = 2 ** i
            padding = int((kernel_size*dilation - dilation)/2)
            in_layer = torch.nn.Conv1d(n_channels, 2*n_channels, kernel_size,
                                       dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            cond_layer = torch.nn.Conv1d(n_mel_channels, 2*n_channels, 1)
            cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')
            self.cond_layers.append(cond_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2*n_channels
            else:
                res_skip_channels = n_channels
            res_skip_layer = torch.nn.Conv1d(n_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, forward_input: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Forward computation of the WN block.

        Args:
            forward_input (Tuple[torch.Tensor, torch.Tensor]): A tuple containing the audio and the spectrogram tensors.

        Returns:
            torch.Tensor: The output tensor of the WN block.
        """
        audio, spect = forward_input
        audio = self.start(audio)  # [batch_size, n_channels, length]

        for i in range(self.n_layers):
            acts = fused_add_tanh_sigmoid_multiply(
                self.in_layers[i](audio),  # [batch_size, 2*n_channels, length]
                self.cond_layers[i](spect),  # [batch_size, 2*n_channels, length]
                torch.IntTensor([self.n_channels]))  # [batch_size, n_channels, length]

            res_skip_acts = self.res_skip_layers[i](acts)  # [batch_size, res_skip_channels, length]
            if i < self.n_layers - 1:
                audio = res_skip_acts[:, :self.n_channels, :] + audio  # [batch_size, n_channels, length]
                skip_acts = res_skip_acts[:, self.n_channels:, :]  # [batch_size, n_channels, length]
            else:
                skip_acts = res_skip_acts  # [batch_size, n_channels, length]

            if i == 0:
                output = skip_acts  # [batch_size, n_channels, length]
            else:
                output = skip_acts + output  # [batch_size, n_channels, length]
        return self.end(output)  # [batch_size, 2*n_in_channels, length]


class WaveGlow(torch.nn.Module):
    """
    WaveGlow model for mel-spectrogram to audio synthesis.

    Args:
        n_mel_channels (int): Number of mel spectrogram channels.
        n_flows (int): Number of normalizing flow steps.
        n_group (int): Number of samples in a group for the squeeze operation.
        n_early_every (int): Every how many flows to output early samples.
        n_early_size (int): Number of early samples to output.
        WN_config (dict): Configuration dictionary for the WN module.

    Example:
        >>> wn_config = {'n_layers': 8, 'n_channels': 256, 'kernel_size': 3}
        >>> waveglow = WaveGlow(n_mel_channels=80, n_flows=12, n_group=8, n_early_every=4, n_early_size=2, WN_config=wn_config)
        >>> spect = torch.randn(2, 80, 100) # [batch_size, n_mel_channels, frames]
        >>> audio = torch.randn(2, 25600) # [batch_size, time]
        >>> z, log_s_list, log_det_W_list = waveglow((spect, audio))
        >>> z.shape
        torch.Size([2, 4, 3200])
    """
    def __init__(self, n_mel_channels: int, n_flows: int, n_group: int, n_early_every: int,
                 n_early_size: int, WN_config: Dict[str, Any]):
        super(WaveGlow, self).__init__()

        self.upsample = torch.nn.ConvTranspose1d(n_mel_channels,
                                                 n_mel_channels,
                                                 1024, stride=256)
        assert(n_group % 2 == 0)
        self.n_flows = n_flows
        self.n_group = n_group
        self.n_early_every = n_early_every
        self.n_early_size = n_early_size
        self.WN = torch.nn.ModuleList()
        self.convinv = torch.nn.ModuleList()

        n_half = int(n_group/2)

        # Set up layers with the right sizes based on how many dimensions
        # have been output already
        n_remaining_channels = n_group
        for k in range(n_flows):
            if k % self.n_early_every == 0 and k > 0:
                n_half = n_half - int(self.n_early_size/2)
                n_remaining_channels = n_remaining_channels - self.n_early_size
            self.convinv.append(Invertible1x1Conv(n_remaining_channels))
            self.WN.append(WN(n_half, n_mel_channels*n_group, **WN_config))
        self.n_remaining_channels = n_remaining_channels  # Useful during inference

    def forward(self, forward_input: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass for training.

        Args:
            forward_input (Tuple[torch.Tensor, torch.Tensor]):
                forward_input[0] = mel_spectrogram: batch x n_mel_channels x frames
                forward_input[1] = audio: batch x time

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]: The latent variable z, list of log scales, and list of log determinants.
        """
        spect, audio = forward_input

        #  Upsample spectrogram to size of audio
        spect = self.upsample(spect)  # [batch_size, n_mel_channels, time]
        assert(spect.size(2) >= audio.size(1))
        if spect.size(2) > audio.size(1):
            spect = spect[:, :, :audio.size(1)]  # [batch_size, n_mel_channels, time]

        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)  # [batch_size, n_groups, n_mel_channels, group_size]
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1)  # [batch_size, n_mel_channels*n_group, n_groups]

        audio = audio.unfold(1, self.n_group, self.n_group).permute(0, 2, 1)  # [batch_size, n_group, n_groups]
        output_audio = []
        log_s_list = []
        log_det_W_list = []

        for k in range(self.n_flows):
            if k % self.n_early_every == 0 and k > 0:
                output_audio.append(audio[:, :self.n_early_size, :])  # [batch_size, n_early_size, n_groups]
                audio = audio[:, self.n_early_size:, :]  # [batch_size, current_channels, n_groups]

            audio, log_det_W = self.convinv[k](audio)  # [batch_size, current_channels, n_groups], []
            log_det_W_list.append(log_det_W)

            n_half = int(audio.size(1)/2)
            audio_0 = audio[:, :n_half, :]  # [batch_size, n_half, n_groups]
            audio_1 = audio[:, n_half:, :]  # [batch_size, n_half, n_groups]

            output = self.WN[k]((audio_0, spect))  # [batch_size, 2*n_half, n_groups]
            log_s = output[:, n_half:, :]  # [batch_size, n_half, n_groups]
            b = output[:, :n_half, :]  # [batch_size, n_half, n_groups]
            audio_1 = torch.exp(log_s)*audio_1 + b  # [batch_size, n_half, n_groups]
            log_s_list.append(log_s)

            audio = torch.cat([audio_0, audio_1], 1)  # [batch_size, current_channels, n_groups]

        output_audio.append(audio)
        return torch.cat(output_audio, 1), log_s_list, log_det_W_list  # [batch_size, n_group, n_groups]

    def infer(self, spect: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """
        Synthesizes audio from a mel spectrogram.

        Args:
            spect (torch.Tensor): The input mel spectrogram. Shape: (batch_size, n_mel_channels, frames)
            sigma (float, optional): Standard deviation of the Gaussian noise. Defaults to 1.0.

        Returns:
            torch.Tensor: The synthesized audio. Shape: (batch_size, time)

        Example:
            >>> waveglow = WaveGlow(80, 12, 8, 4, 2, {'n_layers': 8, 'n_channels': 256, 'kernel_size': 3})
            >>> spect = torch.randn(1, 80, 100)
            >>> audio = waveglow.infer(spect)
            >>> audio.shape
            torch.Size([1, 25600])
        """
        spect = self.upsample(spect)  # [batch_size, n_mel_channels, time]
        # trim conv artifacts. maybe pad spec to kernel multiple
        time_cutoff = self.upsample.kernel_size[0] - self.upsample.stride[0]
        spect = spect[:, :, :-time_cutoff]  # [batch_size, n_mel_channels, trimmed_time]

        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)  # [batch_size, n_groups, n_mel_channels, group_size]
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1)  # [batch_size, n_mel_channels*n_group, n_groups]

        audio = torch.randn(
            spect.size(0), self.n_remaining_channels, spect.size(2),
            dtype=spect.dtype, device=spect.device
        )

        audio = torch.autograd.Variable(sigma*audio)  # [batch_size, n_remaining_channels, n_groups]

        for k in reversed(range(self.n_flows)):
            n_half = int(audio.size(1)/2)
            audio_0 = audio[:, :n_half, :]  # [batch_size, n_half, n_groups]
            audio_1 = audio[:, n_half:, :]  # [batch_size, n_half, n_groups]

            output = self.WN[k]((audio_0, spect))  # [batch_size, 2*n_half, n_groups]
            s = output[:, n_half:, :]  # [batch_size, n_half, n_groups]
            b = output[:, :n_half, :]  # [batch_size, n_half, n_groups]
            audio_1 = (audio_1 - b)/torch.exp(s)  # [batch_size, n_half, n_groups]
            audio = torch.cat([audio_0, audio_1], 1)  # [batch_size, current_channels, n_groups]

            audio = self.convinv[k](audio, reverse=True)  # [batch_size, current_channels, n_groups]

            if k % self.n_early_every == 0 and k > 0:
                z = torch.randn(
                    spect.size(0), self.n_early_size, spect.size(2),
                    dtype=spect.dtype, device=spect.device
                )
                audio = torch.cat((sigma*z, audio), 1)  # [batch_size, current_channels + n_early_size, n_groups]

        audio = audio.permute(0, 2, 1).contiguous().view(audio.size(0), -1).data  # [batch_size, time]
        return audio

    @staticmethod
    def remove_weightnorm(model: torch.nn.Module) -> torch.nn.Module:
        """
        Removes weight normalization from the WaveGlow model for faster inference.

        Args:
            model (torch.nn.Module): The WaveGlow model.

        Returns:
            torch.nn.Module: The model with weight normalization removed.

        Example:
            >>> waveglow = WaveGlow(80, 12, 8, 4, 2, {'n_layers': 8, 'n_channels': 256, 'kernel_size': 3})
            >>> waveglow = WaveGlow.remove_weightnorm(waveglow)
        """
        waveglow = model
        for WN in waveglow.WN:
            WN.start = torch.nn.utils.remove_weight_norm(WN.start)
            WN.in_layers = remove(WN.in_layers)
            WN.cond_layers = remove(WN.cond_layers)
            WN.res_skip_layers = remove(WN.res_skip_layers)
        return waveglow


def remove(conv_list: torch.nn.ModuleList) -> torch.nn.ModuleList:
    """
    Removes weight normalization from a list of convolutional layers.

    Args:
        conv_list (torch.nn.ModuleList): A module list of convolutional layers.

    Returns:
        torch.nn.ModuleList: The module list with weight normalization removed from its elements.

    Example:
        >>> convs = torch.nn.ModuleList([torch.nn.utils.weight_norm(torch.nn.Conv1d(10, 10, 1))])
        >>> convs_unnormed = remove(convs)
    """
    new_conv_list = torch.nn.ModuleList()
    for old_conv in conv_list:
        old_conv = torch.nn.utils.remove_weight_norm(old_conv)
        new_conv_list.append(old_conv)
    return new_conv_list
