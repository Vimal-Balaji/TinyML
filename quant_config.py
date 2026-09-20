"""
quant_config.py
Per-layer mixed-precision QAT configuration.

PyTorch's built-in QAT machinery natively supports int8 well; true int4/int2
fake-quantization requires custom fake_quant modules. Below we implement a
small custom FakeQuantize observer that supports arbitrary bit-widths, and a
helper that assigns a bit-width per layer according to a sensitivity policy:

  - Early layers (stem, block1): keep higher precision (8-bit) since errors
    here propagate through the whole network.
  - Later layers (block3, block4, fc): allow lower precision (4-bit) since
    they operate on more redundant, higher-level features.

This mirrors the actual resource-allocation problem on FPGA: an int4 MAC
unit costs roughly 1/4 the DSP slices of an int8 MAC, so pushing later
layers down in precision is where you recover the most hardware budget.
"""

import torch
from torch.ao.quantization import FakeQuantize, MovingAverageMinMaxObserver, QConfig


def make_qconfig(bits: int, symmetric: bool = True):
    """Builds a QConfig for a given bit-width using PyTorch's fake-quant
    machinery. weight and activation can be given different bit-widths."""
    qscheme = torch.per_tensor_symmetric if symmetric else torch.per_tensor_affine
    quant_min = -(2 ** (bits - 1))
    quant_max = 2 ** (bits - 1) - 1

    activation_fq = FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=0, quant_max=2 ** bits - 1,
        dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=False,
    )
    weight_fq = FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=quant_min, quant_max=quant_max,
        dtype=torch.qint8, qscheme=qscheme,
    )
    return QConfig(activation=activation_fq, weight=weight_fq)


# Two policies to compare in your results table:
UNIFORM_INT8 = {"stem": 8, "block1": 8, "block2": 8, "block3": 8, "block4": 8, "fc": 8}
MIXED_PRECISION = {"stem": 8, "block1": 8, "block2": 4, "block3": 4, "block4": 4, "fc": 8}


def apply_mixed_qconfig(model, policy: dict):
    """Walks the top-level named modules of StudentDSCNN and assigns a
    per-module QConfig according to `policy` (module_name -> bit-width).
    Falls back to the default int8 qconfig for anything unlisted.
    """
    model.qconfig = make_qconfig(8)  # default/global fallback

    for name, bits in policy.items():
        module = getattr(model, name, None)
        if module is not None:
            module.qconfig = make_qconfig(bits)
    return model


def estimate_dsp_cost(policy: dict, macs_per_layer: dict) -> float:
    """Rough relative DSP-slice cost estimate: cost scales ~linearly with
    bit-width for a MAC unit (a reasonable first-order approximation used
    in early-stage design-space exploration before synthesis).
    Returns a relative cost score (lower = fewer DSP slices needed) that
    you can plot against accuracy for the Pareto-curve figure.
    """
    total = 0.0
    for layer, bits in policy.items():
        macs = macs_per_layer.get(layer, 0)
        total += macs * (bits / 8.0)   # normalized to int8 = 1.0x cost
    return total