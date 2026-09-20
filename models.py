"""
models.py
- TeacherCNN: larger, float32-only, used only for knowledge distillation (never deployed).
- StudentDSCNN: small depthwise-separable 1D-CNN — the actual FPGA-target model.
- Quantization stubs (QuantStub/DeQuantStub) are wired into the student for QAT.

Depthwise-separable convolutions are the standard choice for tinyML/FPGA
audio models (see MobileNet / DS-CNN keyword spotting literature) because
they cut multiply-accumulate count roughly 8-9x vs a standard conv, which
maps directly to fewer DSP slices on an FPGA.
"""

import torch
import torch.nn as nn
import torch.ao.quantization as tq


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise conv (per-channel) followed by pointwise 1x1 conv.
    This factorization is the core building block that makes the model
    cheap enough for FPGA deployment."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size, stride=stride,
                                    padding=padding, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm1d(in_ch)
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.act(self.bn2(self.pointwise(x)))
        return x


class TinyStudentDSCNN(nn.Module):
    """Ultra-compact depthwise-separable CNN sized to fit a ~4KB budget
    after int8 quantization (~3.8KB at width=22, leaving headroom for
    per-tensor quant scale/zero-point metadata).

    Architecture is deliberately shallow (stem + 2 DS-conv blocks, no
    intermediate stride-1 blocks) since every extra layer in a model this
    small costs a disproportionate share of the parameter budget. This is
    the actual FPGA-deployment target model; StudentDSCNN (the larger
    version) is kept as a size/accuracy comparison point in your report.
    """

    def __init__(self, n_mfcc=13, num_classes=12, width=22):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        self.stem = nn.Sequential(
            nn.Conv1d(n_mfcc, width, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
        )
        self.block1 = DepthwiseSeparableConv1d(width, width, stride=1)
        self.block2 = DepthwiseSeparableConv1d(width, width * 2, stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(width * 2, num_classes)

    def forward(self, x):
        x = self.quant(x)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        tq.fuse_modules(self.stem, ["0", "1", "2"], inplace=True)
        for block in [self.block1, self.block2]:
            tq.fuse_modules(block, ["depthwise", "bn1"], inplace=True)
            tq.fuse_modules(block, ["pointwise", "bn2"], inplace=True)


def model_size_bytes(model: nn.Module, bits: int = 8) -> float:
    """Estimates on-disk/on-chip size at a given quantization bit-width.
    Rough estimate: ignores BatchNorm params (folded into conv at deploy
    time via fuse_model) and any quant-scale/zero-point metadata overhead,
    which typically adds a few dozen to ~200 bytes depending on how many
    distinct quantized tensors the model has."""
    n_params = sum(p.numel() for p in model.parameters())
    return n_params * (bits / 8.0)


class StudentDSCNN(nn.Module):
    """Small depthwise-separable CNN — the model that actually gets
    quantized and synthesized to the FPGA.

    Input: MFCC features [B, n_mfcc, time]  (treated as 1D sequence over time,
    n_mfcc channels).
    """

    def __init__(self, n_mfcc=13, num_classes=12, width=32):
        super().__init__()
        # QuantStub/DeQuantStub mark the boundaries of the quantized region
        # for PyTorch's QAT machinery.
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        self.stem = nn.Sequential(
            nn.Conv1d(n_mfcc, width, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
        )
        self.block1 = DepthwiseSeparableConv1d(width, width, stride=1)
        self.block2 = DepthwiseSeparableConv1d(width, width * 2, stride=2)
        self.block3 = DepthwiseSeparableConv1d(width * 2, width * 2, stride=1)
        self.block4 = DepthwiseSeparableConv1d(width * 2, width * 4, stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(width * 4, num_classes)

    def forward(self, x):
        x = self.quant(x)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        """Fuse conv+bn+relu where possible — required before QAT for
        realistic accuracy/latency behavior, and mandatory for most
        FPGA toolchains that expect fused ops."""
        tq.fuse_modules(self.stem, ["0", "1", "2"], inplace=True)
        for block in [self.block1, self.block2, self.block3, self.block4]:
            tq.fuse_modules(block, ["depthwise", "bn1"], inplace=True)
            tq.fuse_modules(block, ["pointwise", "bn2"], inplace=True)


class TeacherCNN(nn.Module):
    """Larger, standard (non-separable) CNN used ONLY as a distillation
    teacher during training. Never quantized, never deployed to FPGA —
    runs on GPU/CPU at training time only."""

    def __init__(self, n_mfcc=13, num_classes=12, width=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_mfcc, width, kernel_size=5, stride=2, padding=2), nn.BatchNorm1d(width), nn.ReLU(inplace=True),
            nn.Conv1d(width, width, kernel_size=3, padding=1), nn.BatchNorm1d(width), nn.ReLU(inplace=True),
            nn.Conv1d(width, width * 2, kernel_size=3, stride=2, padding=1), nn.BatchNorm1d(width * 2), nn.ReLU(inplace=True),
            nn.Conv1d(width * 2, width * 2, kernel_size=3, padding=1), nn.BatchNorm1d(width * 2), nn.ReLU(inplace=True),
            nn.Conv1d(width * 2, width * 4, kernel_size=3, stride=2, padding=1), nn.BatchNorm1d(width * 4), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(width * 4, num_classes)

    def forward(self, x):
        x = self.net(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)