"""Single-DCU numeric gates for the YGZP INT8 pre-dispatch and K2 paths.

Run from the repository root after building the HIP extensions::

    python megamoe/dcu_megamoe_opt/tests/check_ygzp_int8_device_numerics.py

The inputs deliberately land on signed INT8 round-to-nearest-even ties.  The
checks also inspect the raw bytes written by the vectorized uint32 stores, not
only the signed tensor values.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import megamoe  # noqa: E402
from megamoe.dcu_megamoe_opt.K2_fused.k2_fused import (  # noqa: E402
    swiglu_quant_int8_channelwise_out,
)


PREDISPATCH_HIDDEN = 4096
TOPK = 8
PREDISPATCH_ROWS = 4
PREDISPATCH_CAPACITY = 6
K2_HIDDEN = 2048
K2_ROWS = 2
INT8_SENTINEL = 91
SCALE_SENTINEL = -123.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _tie_values(length: int) -> torch.Tensor:
    pattern = torch.tensor(
        [-127.0, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 127.0],
        dtype=torch.float32,
    )
    _require(length % pattern.numel() == 0, "tie pattern must evenly tile the row")
    return pattern.repeat(length // pattern.numel())


def _tie_quant(length: int) -> torch.Tensor:
    pattern = torch.tensor(
        [-127, -2, -2, 0, 0, 2, 2, 127],
        dtype=torch.int8,
    )
    _require(length % pattern.numel() == 0, "quant pattern must evenly tile the row")
    return pattern.repeat(length // pattern.numel())


def _unsigned_bytes(values: torch.Tensor) -> torch.Tensor:
    return values.to(torch.int16).bitwise_and(0xFF).to(torch.uint8)


def _assert_int8_and_packed_bytes(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    label: str,
) -> None:
    expected_device = expected.to(device=actual.device)
    torch.testing.assert_close(actual, expected_device, rtol=0.0, atol=0.0)

    actual_bytes = actual.contiguous().view(torch.uint8).cpu()
    expected_bytes = _unsigned_bytes(expected.contiguous())
    torch.testing.assert_close(actual_bytes, expected_bytes, rtol=0.0, atol=0.0)

    first_word = actual_bytes.reshape(-1, 4)[0].tolist()
    expected_first_word = expected_bytes.reshape(-1, 4)[0].tolist()
    _require(
        first_word == expected_first_word,
        f"{label} first packed uint32 bytes differ: "
        f"actual={first_word}, expected={expected_first_word}",
    )


def _assert_scale(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    label: str,
) -> None:
    torch.testing.assert_close(
        actual.float().cpu(),
        expected.float().cpu(),
        rtol=1.0e-6,
        atol=0.0,
        msg=lambda message: f"{label}: {message}",
    )


@torch.inference_mode()
def check_predispatch(device: torch.device) -> None:
    tie = _tie_values(PREDISPATCH_HIDDEN)
    real_x = torch.stack((tie, torch.zeros_like(tie), tie * 2.0, tie * 0.5))
    x = torch.full(
        (PREDISPATCH_CAPACITY, PREDISPATCH_HIDDEN),
        37.0,
        dtype=torch.bfloat16,
        device=device,
    )
    x[:PREDISPATCH_ROWS].copy_(real_x.to(dtype=torch.bfloat16, device=device))

    topk_idx = torch.full(
        (PREDISPATCH_CAPACITY, TOPK),
        -77,
        dtype=torch.int32,
        device=device,
    )
    expected_topk_idx = torch.arange(
        PREDISPATCH_ROWS * TOPK,
        dtype=torch.int32,
    ).reshape(PREDISPATCH_ROWS, TOPK)
    topk_idx[:PREDISPATCH_ROWS].copy_(expected_topk_idx.to(device=device))

    topk_weights = torch.full(
        (PREDISPATCH_CAPACITY, TOPK),
        -3.0,
        dtype=torch.bfloat16,
        device=device,
    )
    expected_topk_weights_bf16 = (
        torch.arange(1, PREDISPATCH_ROWS * TOPK + 1, dtype=torch.float32)
        .reshape(PREDISPATCH_ROWS, TOPK)
        .div(64.0)
        .to(torch.bfloat16)
    )
    topk_weights[:PREDISPATCH_ROWS].copy_(
        expected_topk_weights_bf16.to(device=device)
    )

    out_int8 = torch.full(
        (PREDISPATCH_CAPACITY, PREDISPATCH_HIDDEN),
        INT8_SENTINEL,
        dtype=torch.int8,
        device=device,
    )
    out_scale = torch.full(
        (PREDISPATCH_CAPACITY,),
        SCALE_SENTINEL,
        dtype=torch.float32,
        device=device,
    )
    out_topk_idx = torch.full(
        (PREDISPATCH_CAPACITY, TOPK),
        -999,
        dtype=torch.int64,
        device=device,
    )
    out_topk_weights = torch.full(
        (PREDISPATCH_CAPACITY, TOPK),
        -99.25,
        dtype=torch.float32,
        device=device,
    )

    quant_view, scale_view, topk_idx_view, topk_weights_view = (
        megamoe.mega_moe_pre_dispatch_int8(
            x,
            topk_idx,
            topk_weights,
            out_int8,
            out_scale,
            out_topk_idx,
            out_topk_weights,
            num_tokens=PREDISPATCH_ROWS,
        )
    )
    torch.cuda.synchronize(device)

    _require(
        tuple(quant_view.shape) == (PREDISPATCH_ROWS, PREDISPATCH_HIDDEN),
        f"pre-dispatch quant view has wrong shape: {tuple(quant_view.shape)}",
    )
    _require(
        tuple(scale_view.shape) == (PREDISPATCH_ROWS,),
        f"pre-dispatch scale view has wrong shape: {tuple(scale_view.shape)}",
    )
    _require(quant_view.data_ptr() == out_int8.data_ptr(), "quant view lost output storage")
    _require(scale_view.data_ptr() == out_scale.data_ptr(), "scale view lost output storage")
    _require(
        topk_idx_view.data_ptr() == out_topk_idx.data_ptr(),
        "top-k index view lost output storage",
    )
    _require(
        topk_weights_view.data_ptr() == out_topk_weights.data_ptr(),
        "top-k weight view lost output storage",
    )

    tie_quant = _tie_quant(PREDISPATCH_HIDDEN)
    expected_quant = torch.stack(
        (tie_quant, torch.zeros_like(tie_quant), tie_quant, tie_quant)
    )
    expected_scale = torch.tensor(
        [1.0, 1.0e-4 / 127.0, 2.0, 0.5],
        dtype=torch.float32,
    )
    _assert_int8_and_packed_bytes(
        quant_view,
        expected_quant,
        label="pre-dispatch",
    )
    _assert_scale(scale_view, expected_scale, label="pre-dispatch scale")
    torch.testing.assert_close(
        topk_idx_view.cpu(),
        expected_topk_idx.to(torch.int64),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        topk_weights_view.cpu(),
        expected_topk_weights_bf16.float(),
        rtol=0.0,
        atol=0.0,
    )

    _require(
        bool(torch.all(out_int8[PREDISPATCH_ROWS:] == INT8_SENTINEL).item()),
        "pre-dispatch wrote past num_tokens in the capacity-6 quant buffer",
    )
    _require(
        bool(torch.all(out_scale[PREDISPATCH_ROWS:] == SCALE_SENTINEL).item()),
        "pre-dispatch wrote past num_tokens in the capacity-6 scale buffer",
    )
    _require(
        bool(torch.all(out_topk_idx[PREDISPATCH_ROWS:] == -999).item()),
        "pre-dispatch wrote past num_tokens in the capacity-6 top-k index buffer",
    )
    _require(
        bool(torch.all(out_topk_weights[PREDISPATCH_ROWS:] == -99.25).item()),
        "pre-dispatch wrote past num_tokens in the capacity-6 top-k weight buffer",
    )


@torch.inference_mode()
def check_k2(device: torch.device, *, fast_math: bool) -> None:
    tie = _tie_values(K2_HIDDEN)
    gate = torch.full((K2_ROWS, K2_HIDDEN), 128.0, dtype=torch.float32)
    up = torch.stack((tie / 128.0, torch.zeros_like(tie)))
    x = torch.cat((gate, up), dim=1).to(dtype=torch.bfloat16, device=device)

    quant_guard = 32
    quant_storage = torch.full(
        (quant_guard + K2_ROWS * K2_HIDDEN + quant_guard,),
        INT8_SENTINEL,
        dtype=torch.int8,
        device=device,
    )
    out_int8 = quant_storage[
        quant_guard : quant_guard + K2_ROWS * K2_HIDDEN
    ].view(K2_ROWS, K2_HIDDEN)

    scale_guard = 2
    scale_storage = torch.full(
        (scale_guard + K2_ROWS + scale_guard,),
        SCALE_SENTINEL,
        dtype=torch.float32,
        device=device,
    )
    out_scale = scale_storage[scale_guard : scale_guard + K2_ROWS]
    unused_out_bf16 = torch.empty(0, dtype=torch.bfloat16, device=device)

    quant_view, scale_view = swiglu_quant_int8_channelwise_out(
        x,
        None,
        out_int8,
        out_scale,
        unused_out_bf16,
        num_per_channels=K2_HIDDEN,
        output_bf16=False,
        clamp_value=None,
        fast_math=fast_math,
    )
    torch.cuda.synchronize(device)

    mode = f"K2 fast_math={fast_math}"
    _require(quant_view is out_int8, f"{mode} did not return the supplied quant view")
    _require(tuple(scale_view.shape) == (1, K2_ROWS), f"{mode} scale view shape is wrong")
    _require(scale_view.data_ptr() == out_scale.data_ptr(), f"{mode} scale view lost storage")
    _require(
        quant_view.untyped_storage().data_ptr() == quant_storage.untyped_storage().data_ptr(),
        f"{mode} quant view lost backing storage",
    )
    _require(
        scale_view.untyped_storage().data_ptr() == scale_storage.untyped_storage().data_ptr(),
        f"{mode} scale view lost backing storage",
    )

    tie_quant = _tie_quant(K2_HIDDEN)
    expected_quant = torch.stack((tie_quant, torch.zeros_like(tie_quant)))
    expected_scale = torch.tensor(
        [1.0, 1.0e-4 / 127.0],
        dtype=torch.float32,
    )
    _assert_int8_and_packed_bytes(quant_view, expected_quant, label=mode)
    _assert_scale(scale_view.reshape(-1), expected_scale, label=f"{mode} scale")

    _require(
        bool(torch.all(quant_storage[:quant_guard] == INT8_SENTINEL).item()),
        f"{mode} overwrote the quant prefix guard",
    )
    _require(
        bool(torch.all(quant_storage[-quant_guard:] == INT8_SENTINEL).item()),
        f"{mode} overwrote the quant suffix guard",
    )
    _require(
        bool(torch.all(scale_storage[:scale_guard] == SCALE_SENTINEL).item()),
        f"{mode} overwrote the scale prefix guard",
    )
    _require(
        bool(torch.all(scale_storage[-scale_guard:] == SCALE_SENTINEL).item()),
        f"{mode} overwrote the scale suffix guard",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate YGZP signed INT8 device quantization numerics on one DCU"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="visible CUDA/HIP device index to use (default: 0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if getattr(torch.version, "hip", None) is None:
        raise RuntimeError("this numeric gate requires a HIP-enabled PyTorch build")
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA/HIP device is available")
    if args.device < 0 or args.device >= torch.cuda.device_count():
        raise ValueError(
            f"--device={args.device} is outside [0, {torch.cuda.device_count()})"
        )

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    check_predispatch(device)
    print("[PASS] pre-dispatch hidden=4096 topk=8 RNE/scale/topk/packed-byte gate")

    for fast_math in (True, False):
        check_k2(device, fast_math=fast_math)
        print(
            "[PASS] K2 hidden=2048 gate=128 tie/zero RNE/scale/view/storage/"
            f"packed-byte gate (fast_math={fast_math})"
        )

    print("YGZP INT8 device numeric gates passed")


if __name__ == "__main__":
    main()
