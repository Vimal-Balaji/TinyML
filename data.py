"""
data.py
Google Speech Commands (v0.02) data pipeline for TinyML keyword spotting.

Downloads the dataset via torchaudio, extracts MFCC features (a standard,
lightweight feature representation used in real keyword-spotting hardware,
e.g. Cortex-M / FPGA soft-core pipelines), and returns PyTorch DataLoaders.

We use a reduced label set (the standard "10+unknown+silence" split used in
the original TensorFlow Speech Commands benchmark) to keep the problem
tractable for an FPGA-class model while still being far harder than MNIST:
- variable-length, noisy, real-world audio
- 12-way classification instead of 10-way trivial digits
- requires an on-device feature extraction step (MFCC) before the model
"""

import os
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from torchaudio.datasets import SPEECHCOMMANDS

# ----------------------------------------------------------------------
# Label set: 10 keywords + "unknown" (catch-all for other words) + "silence"
# This mirrors the classic TF Speech Commands benchmark split.
# ----------------------------------------------------------------------
TARGET_LABELS = [
    "yes", "no", "up", "down", "left", "right",
    "on", "off", "stop", "go"
]
ALL_LABELS = TARGET_LABELS + ["unknown", "silence"]
LABEL_TO_IDX = {label: i for i, label in enumerate(ALL_LABELS)}
NUM_CLASSES = len(ALL_LABELS)

SAMPLE_RATE = 16000
N_MFCC = 13          # standard for KWS on constrained hardware
N_FFT = 400           # 25 ms window at 16kHz
HOP_LENGTH = 160       # 10 ms hop
FIXED_LENGTH = 16000    # pad/crop every clip to exactly 1 second


class SubsetSC(SPEECHCOMMANDS):
    """Wraps torchaudio's SPEECHCOMMANDS to expose train/val/test splits
    using the official validation_list.txt / testing_list.txt files."""

    def __init__(self, root, subset: str):
        super().__init__(root=root, download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as f:
                return [
                    os.path.normpath(os.path.join(self._path, line.strip()))
                    for line in f
                ]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = set(load_list("validation_list.txt") + load_list("testing_list.txt"))
            self._walker = [w for w in self._walker if os.path.normpath(w) not in excludes]
        else:
            raise ValueError(f"Unknown subset {subset}")

    def __getitem__(self, n):
        # Overridden to bypass torchaudio.load() -> TorchCodec, which
        # requires a working FFmpeg install. We read the .wav directly
        # with soundfile instead (no FFmpeg dependency).
        fileid = self._walker[n]
        relpath = os.path.relpath(fileid, self._path)
        label, filename = os.path.split(relpath)
        speaker_id = filename.split("_")[0] if "_" in filename else "unknown"

        data, sample_rate = sf.read(fileid, dtype="float32", always_2d=True)
        # sf returns [time, channels] -> torchaudio convention is [channels, time]
        waveform = torch.from_numpy(data.T)
        return waveform, sample_rate, label, speaker_id, 0


def label_to_index(word: str) -> int:
    if word in LABEL_TO_IDX:
        return LABEL_TO_IDX[word]
    return LABEL_TO_IDX["unknown"]


class MFCCKeywordDataset(Dataset):
    """Wraps SubsetSC and converts each raw waveform to a fixed-size
    MFCC feature map suitable for a small 1D/2D CNN."""

    def __init__(self, subset: str, root: str = "./data"):
        self.dataset = SubsetSC(root=root, subset=subset)
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            melkwargs={"n_fft": N_FFT, "hop_length": HOP_LENGTH, "n_mels": 40},
        )

    def __len__(self):
        return len(self.dataset)

    def _fix_length(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: [1, T]
        length = waveform.shape[-1]
        if length < FIXED_LENGTH:
            pad = FIXED_LENGTH - length
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, :FIXED_LENGTH]
        return waveform

    def __getitem__(self, idx):
        waveform, sr, label, *_ = self.dataset[idx]
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        waveform = self._fix_length(waveform)
        features = self.mfcc(waveform)          # [1, n_mfcc, time]
        features = features.squeeze(0)            # [n_mfcc, time]
        target = label_to_index(label)
        return features, target


def collate_fn(batch):
    features = torch.stack([b[0] for b in batch])      # [B, n_mfcc, time]
    targets = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return features, targets


def get_dataloaders(root="./data", batch_size=128, num_workers=2):
    train_ds = MFCCKeywordDataset("training", root)
    val_ds = MFCCKeywordDataset("validation", root)
    test_ds = MFCCKeywordDataset("testing", root)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # quick smoke test
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=8)
    x, y = next(iter(train_loader))
    print("Feature batch shape:", x.shape)   # [B, n_mfcc, time]
    print("Label batch shape:", y.shape)
    print("Num classes:", NUM_CLASSES)