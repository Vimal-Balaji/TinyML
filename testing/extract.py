"""
extract_single_features.py

Extracts MFCC features from ONE audio file, using the exact same
preprocessing settings as data.py (so the numbers match what your trained
model actually saw during training). Prints:
  1. The raw feature values (Python list form)
  2. A ready-to-paste C float array, for hardcoding into an Arduino sketch

Usage:
    python extract_single_features.py path/to/audio.wav

Notes:
- Input can be any format soundfile supports (wav is most common/safe).
- Any sample rate is fine; it's resampled to 16kHz internally.
- Output shape is fixed at [13, 101] (13 MFCC coefficients x 101 time
  frames) as long as the clip is treated as 1 second (padded/cropped),
  matching the FIXED_LENGTH used during training.
"""

import sys
import torch
import torchaudio
import soundfile as sf

# ---- Must match data.py exactly ----
SAMPLE_RATE = 16000
N_MFCC = 13
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 40
FIXED_LENGTH = 16000  # 1 second at 16kHz


def load_audio(path: str) -> torch.Tensor:
    """Loads audio via soundfile (avoids the torchaudio/TorchCodec/FFmpeg
    dependency issue) and returns a [1, T] float32 waveform tensor."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)  # [channels, time]
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # collapse to mono
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    return waveform


def fix_length(waveform: torch.Tensor, length: int = FIXED_LENGTH) -> torch.Tensor:
    """Pads with zeros or crops so every clip is exactly `length` samples,
    matching how the training dataset normalizes clip length."""
    current = waveform.shape[-1]
    if current < length:
        pad = length - current
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    else:
        waveform = waveform[:, :length]
    return waveform


def extract_mfcc(waveform: torch.Tensor) -> torch.Tensor:
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        melkwargs={"n_fft": N_FFT, "hop_length": HOP_LENGTH, "n_mels": N_MELS},
    )
    features = mfcc_transform(waveform)   # [1, n_mfcc, time]
    return features.squeeze(0)             # [n_mfcc, time]


def print_raw(features: torch.Tensor):
    print("\n=== Raw MFCC feature values ===")
    print(f"Shape: {tuple(features.shape)}  (n_mfcc x time_frames)")
    print(f"Total values: {features.numel()}")
    print()
    # Print the complete nested array in Python list format.
    rounded = [[round(v, 6) for v in row] for row in features.tolist()]
    # repr() outputs the complete list; any apparent truncation is from the
    # terminal's scrollback/display limit, not Python.
    sys.stdout.write(repr(rounded) + "\n")


def print_c_array(features: torch.Tensor, var_name: str = "mfcc_features"):
    flat = features.flatten().tolist()
    n_mfcc, n_frames = features.shape

    print("\n=== C array for Arduino (paste directly into your sketch) ===")
    print(f"// Shape: [{n_mfcc}][{n_frames}]  (row-major: coefficient, then time frame)")
    print(f"// Total values: {len(flat)}")
    print(f"#define MFCC_N_COEFFS {n_mfcc}")
    print(f"#define MFCC_N_FRAMES {n_frames}")
    print(f"const float {var_name}[{len(flat)}] = {{")

    # print 8 values per line for readability
    line = []
    for i, v in enumerate(flat):
        line.append(f"{v:.6f}f")
        if len(line) == 8 or i == len(flat) - 1:
            comma = "," if i != len(flat) - 1 else ""
            print("  " + ", ".join(line) + comma)
            line = []
    print("};")


def main():

    audio_path = r"data\SpeechCommands\up\0a2b400e_nohash_2.wav"
    
    waveform = load_audio(audio_path)
    waveform = fix_length(waveform)
    features = extract_mfcc(waveform)
    print(features.shape)

    print(f"Loaded: {audio_path}")
    print(f"Waveform length after fix: {waveform.shape[-1]} samples "
          f"({waveform.shape[-1] / SAMPLE_RATE:.2f} sec)")

    print_raw(features)
    print_c_array(features)


if __name__ == "__main__":
    main()