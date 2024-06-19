# Copyright (C) 2023 Katsuya Iida. All rights reserved.

from typing import Tuple, List, Optional
from voice100_pinktrombone import PinkTrombone
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.nn import init
try:
    import pytorch_lightning as pl
except ImportError:
    class pl:
        class LightningModule:
            pass

        class Callback:
            pass
import random
import numpy as np


class ResNet1d(nn.Module):
    def __init__(
        self,
        n_channels,
        kernel_size: int = 7,
        padding: str = 'valid',
        dilation: int = 1
    ) -> None:
        super().__init__()
        assert padding in ['valid', 'same']
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self._padding_size = (kernel_size // 2) * dilation
        self.conv0 = nn.Conv1d(
            n_channels,
            n_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation)
        self.conv1 = nn.Conv1d(
            n_channels,
            n_channels,
            kernel_size=1)

    def forward(self, input):
        y = input
        x = self.conv0(input)
        x = F.elu(x)
        x = self.conv1(x)
        if self.padding == 'valid':
            y = y[:, :, self._padding_size:-self._padding_size]
        x += y
        x = F.elu(x)
        return x


class ResNet2d(nn.Module):
    def __init__(
        self,
        n_channels: int,
        factor: int,
        stride: Tuple[int, int]
    ) -> None:
        # https://arxiv.org/pdf/2005.00341.pdf
        # The original paper uses layer normalization, but here
        # we use batch normalization.
        super().__init__()
        self.conv0 = nn.Conv2d(
            n_channels,
            n_channels,
            kernel_size=(3, 3),
            padding='same')
        self.bn0 = nn.BatchNorm2d(
            n_channels
        )
        self.conv1 = nn.Conv2d(
            n_channels,
            factor * n_channels,
            kernel_size=(stride[0] + 2, stride[1] + 2),
            stride=stride)
        self.bn1 = nn.BatchNorm2d(
            factor * n_channels
        )
        self.conv2 = nn.Conv2d(
            n_channels,
            factor * n_channels,
            kernel_size=1,
            stride=stride)
        self.bn2 = nn.BatchNorm2d(
            factor * n_channels
        )
        self.pad = nn.ReflectionPad2d([
            (stride[1] + 1) // 2,
            (stride[1] + 2) // 2,
            (stride[0] + 1) // 2,
            (stride[0] + 2) // 2,
        ])
        self.activation = nn.LeakyReLU(0.3)

    def forward(self, input):
        x = self.conv0(input)
        x = self.bn0(x)
        x = self.activation(x)
        x = self.pad(x)
        x = self.conv1(x)
        x = self.bn1(x)

        # shortcut
        y = self.conv2(input)
        y = self.bn2(y)

        x += y
        x = self.activation(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        n_channels: int,
        padding: str,
        stride: int
    ) -> None:
        super().__init__()
        assert padding in ['valid', 'same']
        self.layers = nn.Sequential(
            ResNet1d(n_channels // 2, padding=padding, dilation=1),
            ResNet1d(n_channels // 2, padding=padding, dilation=3),
            ResNet1d(n_channels // 2, padding=padding, dilation=9),
            nn.Conv1d(
                n_channels // 2, n_channels,
                kernel_size=2 * stride,
                padding=(2 * stride) // 2 if padding == 'same' else 0,
                stride=stride),
            nn.ELU(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        n_channels: int,
        padding: str,
        stride: int
    ) -> None:
        super().__init__()
        assert padding in ['valid', 'same']
        self.layers = nn.Sequential(
            nn.ConvTranspose1d(
                n_channels, n_channels // 2,
                kernel_size=2 * stride,
                padding=(2 * stride) // 2 if padding == 'same' else 0,
                stride=stride),
            nn.ELU(),
            ResNet1d(n_channels // 2, padding=padding, dilation=1),
            ResNet1d(n_channels // 2, padding=padding, dilation=3),
            ResNet1d(n_channels // 2, padding=padding, dilation=9),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)


class EncoderDecoder(nn.Module):
    def __init__(self, in_channels, out_channels: int, hidden_dim: int, padding):
        super().__init__()
        assert padding in ['valid', 'same']
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, out_channels, kernel_size=1),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)


class Articulator(nn.Module):
    def __init__(self, num_sections, sample_rate):
        super().__init__()
        self.pinktrombone = PinkTrombone(num_sections)
        self.input_dim = num_sections + 5
        self.sample_rate = sample_rate

    def forward(self, inputs: torch.Tensor):
        x = torch.transpose(inputs.detach().cpu(), 1, 2)
        arr = []
        for i in range(x.shape[0]):
            y = self.pinktrombone(x[i])
            arr.append(y)
        x = np.stack(arr)
        f = np.isnan(x)
        if np.any(f).item():
            x = np.where(f, 0, x)
        f = np.isinf(x)
        if np.any(f).item():
            x = np.where(f, 0, x)
        x = torch.from_numpy(x).to(dtype=inputs.dtype, device=inputs.device)
        x = torchaudio.functional.resample(x, self.pinktrombone.sample_rate, self.sample_rate)
        return x


class Decoder(nn.Module):
    def __init__(self, n_channels: int, padding):
        super().__init__()
        assert padding in ['valid', 'same']
        # self.layers = nn.Sequential(
        #     nn.Conv1d(16 * n_channels, 16 * n_channels, kernel_size=7, padding=padding),
        #     nn.ELU(),
        #     DecoderBlock(16 * n_channels, padding=padding, stride=8),
        #     DecoderBlock(8 * n_channels, padding=padding, stride=5),
        #     DecoderBlock(4 * n_channels, padding=padding, stride=4),
        #     DecoderBlock(2 * n_channels, padding=padding),
        #     nn.Conv1d(n_channels, 1, kernel_size=7, padding=padding),
        #     nn.Tanh(),
        # )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)


class ResidualVectorQuantizer(nn.Module):
    weight: torch.Tensor
    running_mean: torch.Tensor
    code_count: torch.Tensor

    def __init__(
        self,
        num_quantizers: int,
        num_embeddings: int,
        embedding_dim: int,
        decay: float = 0.99,
        code_replace_threshold: float = 0.0001,
        eps: float = 1e-10,
    ) -> None:
        super().__init__()
        self.num_quantizers = num_quantizers
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.register_buffer("weight", torch.empty(num_quantizers, num_embeddings, embedding_dim))
        self.register_buffer("running_mean", torch.empty(num_quantizers, num_embeddings, embedding_dim))
        self.register_buffer("code_count", torch.empty(num_quantizers, num_embeddings))
        self.decay = decay
        self.eps = eps
        self.code_replace_threshold = code_replace_threshold
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.uniform_(self.weight)
        self.running_mean[:] = self.weight
        init.ones_(self.code_count)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        # input: [..., chennel]
        if self.training:
            # Enabling bitrate scalability with quantizer dropout
            n = random.randrange(1, self.num_quantizers)
        else:
            n = self.num_quantizers
        codes = []
        r = input.type_as(self.running_mean).detach()
        with torch.no_grad():
            for i in range(n):
                w = self.weight[i]
                # r: [..., num_embeddings]
                dist = torch.cdist(r, w)
                k = torch.argmin(dist, axis=-1)
                codes.append(k)
                self._update_averages(i, r, k)
                r = r - F.embedding(k, w)
        quantized = input - r
        commitment_loss = torch.mean(torch.square(input - quantized.detach()))
        self.weight.data[:] = self.running_mean / torch.unsqueeze(self.eps + self.code_count, axis=-1)
        return quantized, torch.stack(codes, input.ndim - 1), commitment_loss

    def dequantize(self, input: torch.Tensor, n: Optional[int] = None) -> torch.Tensor:
        # input: [batch_size, length, num_quantizers]
        if n is None:
            n = input.shape[-1]
        assert 0 < n <= self.num_quantizers
        res = 0
        with torch.no_grad():
            for i in range(n):
                k = input[:, :, i]
                w = self.weight[i]
                res += F.embedding(k, w)
        return res

    def _update_averages(self, i: int, r: torch.Tensor, k: torch.Tensor) -> None:
        # https://arxiv.org/pdf/1906.00446.pdf
        # Generating Diverse High-Fidelity Images with VQ-VAE-2
        # 2.1 Vector Quantized Variational AutoEncode

        # k: [...]
        one_hot_k = F.one_hot(torch.flatten(k), self.num_embeddings).type_as(self.code_count)
        code_count_update = torch.mean(one_hot_k, axis=0)
        self.code_count[i].lerp_(code_count_update, 1 - self.decay)

        # r: [..., embedding_dim]
        r = r.reshape(-1, self.embedding_dim)
        running_mean_update = (one_hot_k.T @ r) / r.shape[0]
        self.running_mean[i].lerp_(running_mean_update, 1 - self.decay)

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def replace_vectors(self) -> int:
        # https://arxiv.org/pdf/2107.03312.pdf
        # C. Residual Vector Quantizer:

        # The original paper replaces with an input frame randomly
        # sampled within the current batch.
        # Here we replace with random average of running mean instead.
        num_replaced = torch.sum(self.code_count < self.code_replace_threshold).item()
        if num_replaced > 0:
            for i in range(self.num_quantizers):
                mask = self.code_count[i] < self.code_replace_threshold
                # mask: [num_quantizers, num_embeddings]
                w = torch.rand_like(self.code_count[i])
                w /= torch.sum(w)
                self.running_mean[i, mask] = w.type_as(self.running_mean) @ self.running_mean[i]
                self.code_count[i, mask] = w.type_as(self.code_count) @ self.code_count[i]

        return num_replaced

    @torch.no_grad()
    def calc_entropy(self) -> float:
        p = self.code_count / (self.eps + torch.sum(self.code_count, axis=-1, keepdim=True))
        return -torch.sum(torch.log(p) * p).item() / self.num_quantizers


class WaveDiscriminator(nn.Module):
    r"""MelGAN discriminator from https://arxiv.org/pdf/1910.06711.pdf
    """
    def __init__(self, resolution: int = 1, n_channels: int = 4) -> None:
        super().__init__()
        assert resolution >= 1
        if resolution == 1:
            self.avg_pool = nn.Identity()
        else:
            self.avg_pool = nn.AvgPool1d(resolution * 2, stride=resolution)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.layers = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(1, n_channels, kernel_size=15, padding=7)),
            nn.utils.weight_norm(nn.Conv1d(n_channels, 4 * n_channels, kernel_size=41, stride=4, padding=20, groups=4)),
            nn.utils.weight_norm(nn.Conv1d(4 * n_channels, 16 * n_channels, kernel_size=41, stride=4, padding=20, groups=16)),
            nn.utils.weight_norm(nn.Conv1d(16 * n_channels, 64 * n_channels, kernel_size=41, stride=4, padding=20, groups=64)),
            nn.utils.weight_norm(nn.Conv1d(64 * n_channels, 256 * n_channels, kernel_size=41, stride=4, padding=20, groups=256)),
            nn.utils.weight_norm(nn.Conv1d(256 * n_channels, 256 * n_channels, kernel_size=5, padding=2)),
            nn.utils.weight_norm(nn.Conv1d(256 * n_channels, 1, kernel_size=3, padding=1)),
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.avg_pool(x)
        feats = []
        for layer in self.layers[:-1]:
            x = layer(x)
            feats.append(x)
            x = self.activation(x)
        feats.append(self.layers[-1](x))
        return feats


class STFTDiscriminator(nn.Module):
    r"""STFT-based discriminator from https://arxiv.org/pdf/2107.03312.pdf
    """
    def __init__(
        self, n_fft: int = 1024, hop_length: int = 256,
        n_channels: int = 32
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        n = n_fft // 2 + 1
        for _ in range(6):
            n = (n - 1) // 2 + 1
        self.layers = nn.Sequential(
            nn.Conv2d(1, n_channels, kernel_size=7, padding='same'),
            nn.LeakyReLU(0.3, inplace=True),
            ResNet2d(n_channels, 2, stride=(2, 1)),
            ResNet2d(2 * n_channels, 2, stride=(2, 2)),
            ResNet2d(4 * n_channels, 1, stride=(2, 1)),
            ResNet2d(4 * n_channels, 2, stride=(2, 2)),
            ResNet2d(8 * n_channels, 1, stride=(2, 1)),
            ResNet2d(8 * n_channels, 2, stride=(2, 2)),
            nn.Conv2d(16 * n_channels, 1, kernel_size=(n, 1))
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.shape[1] == 1
        # input: [batch, channel, sequence]
        x = torch.squeeze(input, 1).to(torch.float32)  # torch.stft() doesn't accept float16
        x = torch.stft(x, self.n_fft, self.hop_length, normalized=True, onesided=True, return_complex=True)
        x = torch.abs(x)
        x = torch.unsqueeze(x, dim=1)
        x = self.layers(x)
        return x


class ReconstructionLoss(nn.Module):
    """Reconstruction loss from https://arxiv.org/pdf/2107.03312.pdf
    but uses STFT instead of mel-spectrogram
    """
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, input, target):
        loss = 0
        input = input.to(torch.float32)
        target = target.to(torch.float32)
        for i in range(6, 12):
            s = 2 ** i
            alpha = (s / 2) ** 0.5
            # We use STFT instead of 64-bin mel-spectrogram as n_fft=64 is too small
            # for 64 bins.
            x = torch.stft(input, n_fft=s, hop_length=s // 4, win_length=s, normalized=True, onesided=True, return_complex=True)
            x = torch.abs(x)
            y = torch.stft(target, n_fft=s, hop_length=s // 4, win_length=s, normalized=True, onesided=True, return_complex=True)
            y = torch.abs(y)
            if x.shape[-1] > y.shape[-1]:
                x = x[:, :, :y.shape[-1]]
            elif x.shape[-1] < y.shape[-1]:
                y = y[:, :, :x.shape[-1]]
            loss += torch.mean(torch.abs(x - y))
            loss += alpha * torch.mean(torch.square(torch.log(x + self.eps) - torch.log(y + self.eps)))
        return loss / (12 - 6)


class ReconstructionLoss2(nn.Module):
    """Reconstruction loss from https://arxiv.org/pdf/2107.03312.pdf
    """
    def __init__(self, sample_rate, eps=1e-5):
        super().__init__()
        import torchaudio
        self.layers = nn.ModuleList()
        self.alpha = []
        self.eps = eps
        for i in range(6, 12):
            melspec = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=int(2 ** i),
                win_length=int(2 ** i),
                hop_length=int(2 ** i / 4),
                n_mels=64)
            self.layers.append(melspec)
            self.alpha.append((2 ** i / 2) ** 0.5)

    def forward(self, input, target):
        loss = 0
        for alpha, melspec in zip(self.alpha, self.layers):
            x = melspec(input)
            y = melspec(target)
            if x.shape[-1] > y.shape[-1]:
                x = x[:, y.shape[-1]]
            elif x.shape[-1] < y.shape[-1]:
                y = y[:, x.shape[-1]]
            loss += torch.mean(torch.abs(x - y))
            loss += alpha * torch.mean(torch.square(torch.log(x + self.eps) - torch.log(y + self.eps)))
        return loss


def normalize_control(data):
    return torch.concat([
        data[:, 0:1],
        (data[:, 1:2] - 100.0) / 100.0,
        data[:, 2:4],
        data[:, 4:5] / 0.4,
        (data[:, 5:] + 0.1) / 1.6
    ], axis=1)


def unnormalize_control(data):
    return torch.concat([
        data[:, 0:1],
        data[:, 1:2] * 100.0 + 100.0,
        data[:, 2:4],
        data[:, 4:5] * 0.4,
        data[:, 5:] * 1.6 - 0.1
    ], axis=1)


def rest_diameter(n: int) -> np.ndarray:
    diameter = np.arange(n, dtype=np.float64)
    for i in range(n):
        if i < 7 * n / 44 - 0.5:
            diameter[i] = 0.6
        elif i < 12 * n / 44:
            diameter[i] = 1.1
        else:
            diameter[i] = 1.5
    return diameter


def generate_random_control(T: int, n: int, step: int = 20) -> np.ndarray:
    i = np.arange(n)
    p = np.random.rand((T + 2 * step - 1) // step, 10)
    p = np.repeat(p, step, axis=0)
    p = np.mean(np.stack([p[i:p.shape[0] - 20 + i] for i in range(step)]), axis=0)
    p = p[:T]
    d0 = (np.exp(1 - (n - 2 - i[None, :]) ** 2 / 4) / np.exp(1)) * p[:, None, 0]
    d1 = np.exp(1 - (20 * p[:, None, 2] + 10 - i[None, :]) ** 2 / 200) / np.exp(1) * p[:, None, 1]
    d2 = np.exp(1 - (10 * p[:, None, 4] + 20 - i[None, :]) ** 2 / 20) / np.exp(1) * p[:, None, 3]
    x = np.maximum(d0, d1)
    x = np.maximum(x, d2)
    x = 1.5 - x * 1.6
    rd = rest_diameter(n)
    diameter = np.minimum(x, rd)

    data = np.concatenate([
        p[:, 5:],
        diameter
    ], axis=1)
    data[:, 0] = data[:, 0] > 0.3
    data[:, 1] = data[:, 1] * 100 + 100
    data[:, 4] = np.clip(data[:, 4] * 0.6 - 0.1, 0.01, 0.4)

    return data


def dummy_data_(N=44, T=2000, M=100):
    import numpy as np

    X = np.random.normal(size=(T, N + 5))
    w = np.hamming(M)
    w = w / np.sum(w)
    Y = X[M - 1:, :] * w[0]
    for i in range(1, M):
        # print(X[M-1-i:-i, :].shape)
        Y = Y + X[M-1-i:-i, :] * w[i]
    X = Y / np.std(Y)
    del Y

    X[:, 0] = X[:, 0] > -.1  # voiced
    X[:, 1] = np.clip(X[:, 1] * 40 + 160 + (44 - N) * ((220 - 160) / (44 - 38)), 80, 300)  # frequency
    X[:, 2:4] = np.clip(X[:, 2:4] * 0.5 + 0.7, 0.0, 1.0)  # tenseness
    X[:, 3] = X[:, 2] ** 0.8  # loudness
    X[:, 4] = np.clip(X[:, 4] * 0.1 + 0.01, 0.01, 0.2)  # velum
    X[:, 6:-1] = (X[:, 5:-2] + X[:, 6:-1] + X[:, 7:]) / 3.
    X[:, 6:-1] = (X[:, 5:-2] + X[:, 6:-1] + X[:, 7:]) / 3.
    X[:, 6:-1] = (X[:, 5:-2] + X[:, 6:-1] + X[:, 7:]) / 3.
    X[:, 5:] = np.clip(X[:, 5:] + 1.0, 0., 3.)

    return X


def unnormalize_control_(inputs):
    x = torch.transpose(inputs, 1, 2)
    x = x.numpy().astype(np.float64)
    x[:, :, 0] = x[:, :, 0] > 0.0
    x[:, :, 1] = np.clip(x[:, :, 1] * 100. + 150., 90.0, 250.0)
    x[:, :, 2:4] = np.clip(x[:, :, 2:4] + 0.5, 0.0, 1.0)
    x[:, :, 4] = np.clip(x[:, :, 4] + 0.5, 0.01, 0.2)
    x[:, :, 5:] = np.clip(x[:, :, 5:] * 2.0 + 1.0, 0.0, 3.0)
    x = torch.transpose(torch.from_numpy(x).to(inputs.dtype), 1, 2)
    return x


class PinkTromboneModel(pl.LightningModule):
    def __init__(
        self,
        n_mfcc: int = 32,
        hidden_dim: int = 256,
        num_sections: int = 44,
        num_quantizers: int = 8,
        num_embeddings: int = 1024,
        padding: str = "valid",
        batch_size: int = 32,
        sample_rate: int = 22_050,
        segment_length: int = 32270 * 5,
        lr: float = 1e-4,
        b1: float = 0.5,
        b2: float = 0.9,
        dataset: str = 'librispeech'
    ) -> None:
        # https://arxiv.org/pdf/2009.02095.pdf
        # 2. Method
        # SEANet uses Adam with lr=1e-4, beta1=0.5, beta2=0.9
        # batch_size=16
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate, n_mfcc=n_mfcc, log_mels=True,
            melkwargs=dict(n_fft=512, hop_length=256, n_mels=80, center=False))
        self.articulator = Articulator(num_sections, sample_rate)
        self.encoder = EncoderDecoder(n_mfcc, self.articulator.input_dim, hidden_dim, padding)
        self.decoder = EncoderDecoder(self.articulator.input_dim + n_mfcc, 1, hidden_dim, padding)
        self.criterion = nn.MSELoss(reduction="none")
        self.std_alpha = 0.99
        self.register_buffer('running_std', torch.ones(2) * 0.1)
        self.running_std: torch.Tensor

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        optimizer_g = torch.optim.Adam(
            self.encoder.parameters(),
            lr=lr, betas=(b1, b2))
        optimizer_d = torch.optim.Adam(
            self.decoder.parameters(),
            lr=lr, betas=(b1, b2))
        return [optimizer_g, optimizer_d], []

    def training_step(self, batch, batch_idx):
        optimizer_g, optimizer_d = self.optimizers()

        self.running_std[0] = torch.clip(
            self.running_std[0] * self.std_alpha + torch.std(batch) * (1 - self.std_alpha),
            0.01, 1.0)
        targets = self.transform(batch)
        # targets: [batch, channel, sequence]

        # train generator
        self.toggle_optimizer(optimizer_g)
        control = self.encoder(targets)
        control = torch.sigmoid(control)
        length = min(targets.shape[2], control.shape[2])
        estimates = self.decoder(torch.concat([control[:, :, :length], targets[:, :, :length]], axis=1))
        # estimates: [batch, 1, sequence]
        # print(input.shape, output.shape)

        g_loss = torch.mean(estimates)
        self.log("g_loss", g_loss, prog_bar=True)

        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        self.toggle_optimizer(optimizer_d)

        with torch.no_grad():
            control = control.detach()
            unnormalized = unnormalize_control(control)
            x = self.articulator(unnormalized)
            outputs = self.transform(x * self.running_std[0] / self.running_std[1])

            length = min(outputs.shape[2], targets.shape[2])
            error = self.criterion(outputs[:, :, :length], targets[:, :, :length])
            error = torch.mean(error, axis=1, keepdim=True)
            self.log("error", torch.mean(error), prog_bar=True)

        length = min(targets.shape[2], control.shape[2])
        estimates = self.decoder(torch.concat([control[:, :, :length], targets[:, :, :length]], axis=1))

        length = min(estimates.shape[2], error.shape[2])
        d_loss = self.criterion(estimates[:, :, :length], error[:, :, :length])
        d_loss = torch.mean(d_loss)

        self.log("d_loss", d_loss, prog_bar=True)

        with torch.no_grad():
            x = np.stack([
                generate_random_control(targets.shape[2], self.hparams.num_sections).T
                for i in range(control.shape[0])
            ])
            unnormalized = torch.from_numpy(x).to(device=targets.device, dtype=targets.dtype)
            control = normalize_control(unnormalized)
            x = self.articulator(unnormalized)
            self.running_std[1] = torch.clip(
                self.running_std[1] * self.std_alpha + torch.std(x) * (1 - self.std_alpha),
                0.01, 1.0)
            self.log("t_std", self.running_std[0], prog_bar=True)
            self.log("o_std", self.running_std[1], prog_bar=True)
            outputs = self.transform(x)

            length = min(outputs.shape[2], targets.shape[2])
            error = self.criterion(outputs[:, :, :length], targets[:, :, :length])
            error = torch.mean(error, axis=1, keepdim=True)

        length = min(targets.shape[2], control.shape[2])
        estimates = self.decoder(torch.concat([control[:, :, :length], targets[:, :, :length]], axis=1))

        length = min(estimates.shape[2], error.shape[2])
        f_loss = self.criterion(estimates[:, :, :length], error[:, :, :length])
        f_loss = torch.mean(f_loss)

        self.log("f_loss", f_loss, prog_bar=True)

        loss = d_loss + f_loss

        self.manual_backward(loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

    def train_dataloader(self):
        return self._make_dataloader(True)

    def _make_dataloader(self, train: bool):
        import torchaudio

        def collate(examples):
            return torch.stack(examples)

        class VoiceDataset(torch.utils.data.Dataset):
            def __init__(self, dataset, sample_rate, segment_length):
                self._dataset = dataset
                self._sample_rate = sample_rate
                self._segment_length = segment_length

            def __getitem__(self, index):
                import random
                x, sample_rate, *_ = self._dataset[index]
                x = torchaudio.functional.resample(x, sample_rate, self._sample_rate)
                assert x.shape[0] == 1
                x = torch.squeeze(x)
                x *= 0.95 / torch.max(x)
                assert x.dim() == 1
                if x.shape[0] < self._segment_length:
                    x = F.pad(x, [0, self._segment_length - x.shape[0]], "constant")
                pos = random.randint(0, x.shape[0] - self._segment_length)
                x = x[pos:pos + self._segment_length]
                return x

            def __len__(self):
                return len(self._dataset)

        if self.hparams.dataset == 'yesno':
            ds = torchaudio.datasets.YESNO("./data", download=True)
        elif self.hparams.dataset == 'librispeech-dev':
            ds = torchaudio.datasets.LIBRISPEECH("./data", url="dev-clean")
        elif self.hparams.dataset == 'librispeech':
            url = "train-clean-100" if train else "dev-clean"
            ds = torchaudio.datasets.LIBRISPEECH("./data", url=url)
        else:
            raise ValueError()
        ds = VoiceDataset(ds, self.hparams.sample_rate, self.hparams.segment_length)

        loader = torch.utils.data.DataLoader(
            ds, batch_size=self.hparams['batch_size'], shuffle=True,
            collate_fn=collate)
        return loader


class KMeanCodebookInitCallback(pl.Callback):
    def on_fit_start(self, trainer, model):
        # https://arxiv.org/pdf/2107.03312.pdf
        # C. Residual Vector Quantizer
        # run the k-means
        # algorithm on the first training batch and use the learned
        # centroids as initialization
        batch = next(iter(model.train_dataloader()))
        input = batch[:, None, :].to(model.device)
        with torch.no_grad():
            x = torch.flatten(model.encoder(input))
            mean = torch.mean(x, axis=0)
            std = torch.std(x, axis=0)
            torch.nn.init.normal_(model.quantizer.weight, mean=mean, std=std)
        print(f"KMeanCodebookInitCallback {mean} {std}")


def train():
    model = PinkTromboneModel(
        batch_size=3,
        sample_rate=22_500,
        segment_length=32270 * 5,
        padding='same',
        dataset='librispeech')
    trainer = pl.Trainer(
        max_epochs=10000,
        log_every_n_steps=2,
        # precision='16-mixed',
        logger=pl.loggers.CSVLogger("."),
        # logger=pl.loggers.TensorBoardLogger("lightning_logs", name="soundstream"),
        callbacks=[
            pl.callbacks.ModelCheckpoint(save_last=True, every_n_train_steps=5000),
        ],
    )
    trainer.fit(
        model,
        # ckpt_path="lightning_logs/version_0/checkpoints/last.ckpt"
    )

    return model


if __name__ == "__main__":
    train()
