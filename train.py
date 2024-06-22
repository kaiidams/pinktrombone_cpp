# Copyright (C) 2023 Katsuya Iida. All rights reserved.

from voice100_pinktrombone import PinkTrombone
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
try:
    import pytorch_lightning as pl
except ImportError:
    class pl:
        class LightningModule:
            pass

        class Callback:
            pass
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

    def forward(self, inputs: torch.Tensor, output_length: int):
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
        x = np.clip(x, -1.0, 1.0)
        x = torch.from_numpy(x)
        x = torchaudio.functional.resample(x, self.pinktrombone.sample_rate, self.sample_rate)
        if x.shape[1] > output_length:
            x = x[:, :output_length]
        elif x.shape[1] < output_length:
            x = torch.nn.functional.pad(x, pad=[0, output_length - x.shape[1]])
        return x.to(dtype=inputs.dtype, device=inputs.device)


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
        segment_length: int = 640 * 256,
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
    
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        '''Predicts PinkTrombone control data from MFCC audio.
        '''
        # audio: [batch, channel, sequence]
        control = self.encoder(audio)
        control = torch.sigmoid(control)
        return control

    def update_running_std(self, idx: int, audio: torch.Tensor) -> None:
        self.running_std[idx] = torch.clip(
            self.running_std[idx] * self.std_alpha + torch.std(audio) * (1 - self.std_alpha),
            0.01, 1.0)

    def decode(self, control: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        '''Estimates How far they are different, output of PinkTrombone outputs
        for given control and audio.'''
        estimates = self.decoder(torch.concat([control, audio], axis=1))
        return estimates

    def training_step(self, batch, batch_idx):
        optimizer_g, optimizer_d = self.optimizers()

        self.update_running_std(0, batch)
        output_length = batch.shape[1]
        targets = self.transform(batch)
        # targets: [batch, channel, sequence]

        # train generator
        self.toggle_optimizer(optimizer_g)
        control = self.encode(targets)
        estimates = self.decode(control, targets)
        # estimates: [batch, 1, sequence]

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
            x = self.articulator(unnormalized, output_length)
            outputs = self.transform(x * self.running_std[0] / self.running_std[1])

            error = self.criterion(outputs, targets)
            error = torch.mean(error, axis=1, keepdim=True)
            self.log("error", torch.mean(error), prog_bar=True)

        estimates = self.decode(control, targets)

        d_loss = self.criterion(estimates, error)
        d_loss = torch.mean(d_loss)

        self.log("d_loss", d_loss, prog_bar=True)

        with torch.no_grad():
            x = np.stack([
                generate_random_control(targets.shape[2], self.hparams.num_sections).T
                for i in range(control.shape[0])
            ])
            unnormalized = torch.from_numpy(x).to(device=targets.device, dtype=targets.dtype)
            control = normalize_control(unnormalized)
            x = self.articulator(unnormalized, output_length)
            self.update_running_std(1, batch)
            self.log("t_std", self.running_std[0], prog_bar=True)
            self.log("o_std", self.running_std[1], prog_bar=True)
            outputs = self.transform(x)

            error = self.criterion(outputs, targets)
            error = torch.mean(error, axis=1, keepdim=True)

        estimates = self.decode(control, targets)

        f_loss = self.criterion(estimates, error)
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


def train():
    model = PinkTromboneModel(
        batch_size=3,
        sample_rate=22_050,
        segment_length=640 * 256,
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
