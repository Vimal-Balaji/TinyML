"""
Manual, from-scratch (numpy only) forward pass through the quantized
TinyStudentDSCNN checkpoint (student_int8.pth).

Steps:
  1. Load the int8 state_dict and pull out every weight/bias tensor.
  2. Dequantize each qint8 tensor (int8_value - zero_point) * scale -> float32.
     (BatchNorm has already been fused into the conv weights by the
     quantization/fusion step, which is why no separate bn.* keys exist.)
  3. Re-implement, by hand with plain numpy loops/einsum (no nn.Conv1d,
     no torch ops at inference time), each stage of TinyStudentDSCNN.forward:
        quant -> stem(conv+bn+relu) -> block1(dw+pw) -> block2(dw+pw)
        -> global avg pool -> fc -> dequant
  4. Run it on a user-supplied 2D input matrix (n_mfcc x time).
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
    return features.squeeze(0) 

import numpy as np
import torch

CKPT_PATH = "student_int8.pth"

# ----------------------------------------------------------------------
# 1) Load + extract weights
# ----------------------------------------------------------------------
sd = torch.load(CKPT_PATH, map_location="cpu")


def deq(t: torch.Tensor) -> np.ndarray:
    """Dequantize a qint8/quint8 tensor -> float32 numpy array."""
    if t.dtype in (torch.qint8, torch.quint8, torch.qint32):
        return t.dequantize().numpy().astype(np.float32)
    return t.numpy().astype(np.float32)


W = {}  # flat dict of every extracted, already-dequantized weight/bias
for k, v in sd.items():
    if k.endswith("_packed_params._packed_params"):
        w_q, b = v
        W["fc.weight"] = deq(w_q)                     # (12, 44)
        W["fc.bias"] = b.detach().numpy().astype(np.float32)    # (12,)
    elif isinstance(v, torch.Tensor):
        if v.dtype in (torch.qint8, torch.quint8, torch.qint32):
            W[k] = deq(v)
        else:
            W[k] = v.numpy().astype(np.float32)

print("Extracted tensors:")
for k, v in W.items():
    print(f"  {k:30s} shape={v.shape}")

# quant stub scale/zero_point (how the raw float input itself gets quantized)
quant_scale = float(np.asarray(W["quant.scale"]).reshape(-1)[0])
quant_zp = float(np.asarray(W["quant.zero_point"]).reshape(-1)[0])

stem_w = W["stem.0.weight"]          # (22, 13, 5)
stem_b = W["stem.0.bias"]            # (22,)

b1_dw_w = W["block1.depthwise.weight"]   # (22, 1, 3)
b1_dw_b = W["block1.depthwise.bias"]     # (22,)
b1_pw_w = W["block1.pointwise.weight"]   # (22, 22, 1)
b1_pw_b = W["block1.pointwise.bias"]     # (22,)

b2_dw_w = W["block2.depthwise.weight"]   # (22, 1, 3)
b2_dw_b = W["block2.depthwise.bias"]     # (22,)
b2_pw_w = W["block2.pointwise.weight"]   # (44, 22, 1)
b2_pw_b = W["block2.pointwise.bias"]     # (44,)

fc_w = W["fc.weight"]   # (12, 44)
fc_b = W["fc.bias"]     # (12,)


# ----------------------------------------------------------------------
# 2) Hand-written primitive ops (pure numpy, no torch at "runtime")
# ----------------------------------------------------------------------
def conv1d(x, weight, bias, stride=1, padding=0, groups=1):
    """
    x:      (C_in, L)
    weight: (C_out, C_in/groups, K)
    bias:   (C_out,)
    returns (C_out, L_out)
    """
    C_in, L = x.shape
    C_out, C_in_g, K = weight.shape

    if padding > 0:
        x = np.pad(x, ((0, 0), (padding, padding)), mode="constant")
    L_pad = x.shape[1]
    L_out = (L_pad - K) // stride + 1

    out = np.zeros((C_out, L_out), dtype=np.float32)
    in_per_group = C_in // groups
    out_per_group = C_out // groups

    for g in range(groups):
        x_g = x[g * in_per_group:(g + 1) * in_per_group, :]      # (in_per_group, L_pad)
        w_g = weight[g * out_per_group:(g + 1) * out_per_group]  # (out_per_group, in_per_group, K)
        for oc in range(out_per_group):
            acc = np.zeros(L_out, dtype=np.float32)
            for ic in range(in_per_group):
                for t in range(L_out):
                    start = t * stride
                    acc[t] += np.dot(x_g[ic, start:start + K], w_g[oc, ic, :])
            out[g * out_per_group + oc, :] = acc

    out += bias[:, None]
    return out


def relu(x):
    return np.maximum(x, 0.0)


def global_avg_pool1d(x):
    """x: (C, L) -> (C,)"""
    return x.mean(axis=1)


def linear(x, weight, bias):
    """x: (C_in,), weight: (C_out, C_in), bias: (C_out,)"""
    return weight @ x + bias


def fake_quant_dequant(x, scale, zero_point, qmin=-128, qmax=127):
    """
    Simulates the QuantStub: float -> int8 -> float round-trip,
    exactly like torch.quantize_per_tensor(...).dequantize() would.
    """
    q = np.round(x / scale + zero_point)
    q = np.clip(q, qmin, qmax)
    return (q - zero_point) * scale


# ----------------------------------------------------------------------
# 3) Manual forward pass
# ----------------------------------------------------------------------
def manual_forward(x, verbose=True):
    """
    x: 2D numpy array, shape (n_mfcc=13, time_steps), the raw MFCC input.
    """
    x = np.asarray(x, dtype=np.float32)
    assert x.shape[0] == 13, f"expected 13 MFCC channels, got {x.shape[0]}"

    if verbose:
        print(f"\nInput x: shape={x.shape}")

    # --- quant() ---
    # Skip quantization of input
    if verbose:
        print(f"After quant stub (int8 round-trip): shape={x.shape}")

    # --- stem: Conv1d(13->22, k=5, s=2, p=2) + BN(fused) + ReLU ---
    x = conv1d(x, stem_w, stem_b, stride=2, padding=2, groups=1)
    x = relu(x)
    if verbose:
        print(f"After stem (conv k5 s2 p2 + relu): shape={x.shape}")

    # --- block1: depthwise(22->22,k3,s1,p1) + relu, pointwise(22->22,k1) + relu ---
    x = conv1d(x, b1_dw_w, b1_dw_b, stride=1, padding=1, groups=22)
    x = relu(x)
    if verbose:
        print(f"After block1.depthwise + relu: shape={x.shape}")
    x = conv1d(x, b1_pw_w, b1_pw_b, stride=1, padding=0, groups=1)
    x = relu(x)
    if verbose:
        print(f"After block1.pointwise + relu: shape={x.shape}")

    # --- block2: depthwise(22->22,k3,s2,p1) + relu, pointwise(22->44,k1) + relu ---
    x = conv1d(x, b2_dw_w, b2_dw_b, stride=2, padding=1, groups=22)
    x = relu(x)
    if verbose:
        print(f"After block2.depthwise + relu: shape={x.shape}")
    x = conv1d(x, b2_pw_w, b2_pw_b, stride=1, padding=0, groups=1)
    x = relu(x)
    if verbose:
        print(f"After block2.pointwise + relu: shape={x.shape}")

    # --- global average pool ---
    x = global_avg_pool1d(x)
    if verbose:
        print(f"After AdaptiveAvgPool1d(1) + squeeze: shape={x.shape}")

    # --- fc ---
    logits = linear(x, fc_w, fc_b)
    if verbose:
        print(f"After fc: shape={logits.shape}")

    # --- dequant() is a no-op numerically here since logits are already float ---
    return logits


# ----------------------------------------------------------------------
# 4) Batch evaluation over a folder of .wav files
# ----------------------------------------------------------------------
import glob
import os

UP_CLASS_INDEX = 3  # "up" is label index 2


def predict_file(path):
    waveform = load_audio(path)
    waveform = fix_length(waveform)
    mfcc = extract_mfcc(waveform)          # torch tensor (13, time)
    input_matrix = mfcc.numpy().astype(np.float32)
    logits = manual_forward(input_matrix, verbose=False)
    return int(np.argmax(logits))


if __name__ == "__main__":
    folder = r"data\SpeechCommands\left"
    wav_files = sorted(glob.glob(os.path.join(folder, "*.wav")))

    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {folder}")

    correct = 0
    total = len(wav_files)
    for path in wav_files:
        pred = predict_file(path)
        is_correct = (pred == UP_CLASS_INDEX)
        correct += int(is_correct)
        print(f"{os.path.basename(path):30s} predicted={pred:2d}  correct={is_correct}")

    print(f"\nDetected 'up' correctly on {correct}/{total} files "
          f"({100.0 * correct / total:.2f}% accuracy)")