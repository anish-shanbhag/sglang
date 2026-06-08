"""Correctness tests for the FlashInfer cutlass FP8 block-scale fused-MoE backend.

These tests cover the SwiGLU gate/up operand-order fix that the
``flashinfer_cutlass`` MoE backend requires for plain fp8 block-scale
checkpoints.

Background
----------
sglang stores the fused first-FC weight ``w13`` as ``[gate; up]`` (i.e.
``[w1; w3]``) along the intermediate dimension, and the mathematically correct
gated activation is ``silu(gate) * up`` -- exactly what sglang's reference
Triton ``fused_moe`` / native ``silu_and_mul`` path computes.

FlashInfer's cutlass GLU activation, however, computes
``silu(second FC1 half) * (first FC1 half)``. Fed the unmodified ``[gate; up]``
layout it would compute ``silu(up) * gate``, a *different* function whose
per-layer error is small enough to slip past a naive cosine check yet large
enough to compound across layers into incoherent generation.

The backend therefore swaps the two FC1 (``w13``) weight halves and the
matching ``[128]``-block scale rows once at load time, producing ``[up; gate]``,
so the kernel evaluates ``silu(gate) * up``.

Two tests guard this:

1. ``TestW13GateUpSwap`` -- a pure-tensor unit test of the swap permutation
   itself (no GPU required). It documents exactly what the swap does to the
   weight and to its block scales.
2. ``TestFlashInferCutlassFp8BlockScaleMoE`` -- a small end-to-end unit test
   (FlashInfer + Hopper/SM90 required) that runs a few-expert block-scale MoE
   through the cutlass path with the swap applied and asserts it matches the
   reference ``silu(gate) * up`` MoE within an absolute/relative tolerance,
   while the *unswapped* layout does not.

The native ``silu(gate) * up`` block-fp8 MoE reference is reused directly from
the sibling ``test_block_fp8.py`` instead of being re-derived here. Because
``test/manual/quant/`` has no ``__init__.py``, it is imported via a sibling
path insertion.
"""

import os
import sys
import unittest

import torch

from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.utils import get_device_sm
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, os.path.dirname(__file__))
from test_block_fp8 import (  # noqa: E402  native silu(gate)*up block-fp8 MoE reference
    torch_w8a8_block_fp8_moe,
)

_is_cuda = torch.cuda.is_available() and torch.version.cuda

BLOCK_SIZE = [128, 128]


def swap_w13_gate_up_halves(weight: torch.Tensor, scale: torch.Tensor):
    """Swap the two FC1 halves of ``w13`` and the matching block-scale rows.

    Mirrors the ``flashinfer_cutlass`` backend's
    ``process_weights_after_loading`` step: turn the loader's ``[gate; up]``
    layout into the ``[up; gate]`` layout the cutlass GLU activation expects.

    Args:
        weight: ``w13`` weight, shape ``[E, 2 * I, H]`` (``[gate; up]`` along
            dim 1).
        scale: ``w13`` block-scale tensor, shape ``[E, 2 * I / 128, H / 128]``.

    Returns:
        ``(weight, scale)`` re-ordered to ``[up; gate]``.
    """
    half = weight.shape[1] // 2
    swapped_w = torch.cat(
        [weight[:, half:, :], weight[:, :half, :]], dim=1
    ).contiguous()
    s_half = scale.shape[1] // 2
    swapped_s = torch.cat(
        [scale[:, s_half:, :], scale[:, :s_half, :]], dim=1
    ).contiguous()
    return swapped_w, swapped_s


class TestW13GateUpSwap(CustomTestCase):
    """Pure-tensor unit test of the gate/up swap permutation (no GPU)."""

    def test_swap_reorders_gate_up_and_scale_rows(self):
        torch.manual_seed(0)
        E, I, H = 2, 256, 512  # 2*I and H both multiples of 128
        block = 128

        gate = torch.randn(E, I, H)
        up = torch.randn(E, I, H)
        w13 = torch.cat([gate, up], dim=1)  # [gate; up]

        s_rows = (2 * I) // block
        s_cols = H // block
        scale = torch.arange(E * s_rows * s_cols, dtype=torch.float32).reshape(
            E, s_rows, s_cols
        )
        gate_s, up_s = scale.split(s_rows // 2, dim=1)

        swapped_w, swapped_s = swap_w13_gate_up_halves(w13, scale)

        # Weight halves are exactly exchanged: result is [up; gate].
        torch.testing.assert_close(swapped_w[:, :I, :], up)
        torch.testing.assert_close(swapped_w[:, I:, :], gate)

        # Scale rows are exchanged the same way: [up_scale; gate_scale].
        torch.testing.assert_close(swapped_s[:, : s_rows // 2, :], up_s)
        torch.testing.assert_close(swapped_s[:, s_rows // 2 :, :], gate_s)

        # Swap is its own inverse.
        round_trip_w, round_trip_s = swap_w13_gate_up_halves(swapped_w, swapped_s)
        torch.testing.assert_close(round_trip_w, w13)
        torch.testing.assert_close(round_trip_s, scale)


@unittest.skipUnless(_is_cuda, "CUDA is required")
class TestFlashInferCutlassFp8BlockScaleMoE(CustomTestCase):
    """End-to-end correctness of the cutlass FP8 block-scale MoE path.

    Proves the gate/up swap makes the cutlass backend compute
    ``silu(gate) * up`` (matching the reference), and that omitting the swap
    yields a measurably different result.
    """

    # Small DeepSeek-shaped block-scale MoE. Intermediate and hidden are
    # multiples of 128 so the [128, 128] block scales tile evenly.
    M = 64  # tokens
    E = 8  # experts
    TOPK = 2
    HIDDEN = 512
    INTERMEDIATE = 256

    @classmethod
    def setUpClass(cls):
        if not _is_cuda:
            raise unittest.SkipTest("CUDA is required")
        if get_device_sm() != 90:
            raise unittest.SkipTest(
                "FlashInfer cutlass FP8 block-scale MoE requires Hopper (SM90)"
            )
        try:
            from flashinfer.fused_moe import cutlass_fused_moe  # noqa: F401
        except Exception as err:  # pragma: no cover - import guard
            raise unittest.SkipTest(f"FlashInfer cutlass_fused_moe unavailable: {err}")
        torch.set_default_device("cuda")

    def _build_block_scale_moe(self):
        torch.manual_seed(0)
        N, K = self.INTERMEDIATE, self.HIDDEN
        block_n, block_k = BLOCK_SIZE
        finfo = torch.finfo(torch.float8_e4m3fn)
        fp8_max, fp8_min = finfo.max, finfo.min
        factor = 1e-2  # keep accumulation in range, per the in-tree reference

        a = (torch.randn(self.M, K, dtype=torch.bfloat16) / 10).contiguous()

        # w13 == [gate; up], shape [E, 2N, K]; w2 shape [E, K, N].
        w13 = (
            ((torch.rand(self.E, 2 * N, K, dtype=torch.float32) - 0.5) * 2 * fp8_max)
            .clamp(fp8_min, fp8_max)
            .to(torch.float8_e4m3fn)
        )
        w2 = (
            ((torch.rand(self.E, K, N, dtype=torch.float32) - 0.5) * 2 * fp8_max)
            .clamp(fp8_min, fp8_max)
            .to(torch.float8_e4m3fn)
        )
        w13_s = (
            torch.rand(
                self.E, (2 * N) // block_n, K // block_k, dtype=torch.float32
            )
            * factor
        )
        w2_s = (
            torch.rand(self.E, K // block_n, N // block_k, dtype=torch.float32)
            * factor
        )
        score = torch.randn(self.M, self.E, dtype=torch.bfloat16)
        return a, w13, w2, w13_s, w2_s, score

    def _run_cutlass(self, a, w13, w2, w13_s, w2_s, topk_weights, topk_ids):
        from flashinfer.fused_moe import cutlass_fused_moe
        from flashinfer.fused_moe.core import ActivationType

        out = cutlass_fused_moe(
            input=a,
            token_selected_experts=topk_ids.to(torch.int).contiguous(),
            token_final_scales=topk_weights.contiguous(),
            fc1_expert_weights=w13,
            fc2_expert_weights=w2,
            output_dtype=a.dtype,
            quant_scales=[w13_s.contiguous(), w2_s.contiguous()],
            use_deepseek_fp8_block_scale=True,
            tune_max_num_tokens=self.M,
            activation_type=ActivationType.Swiglu,
        )[0]
        return out

    def test_cutlass_matches_reference_with_swap(self):
        a, w13, w2, w13_s, w2_s, score = self._build_block_scale_moe()

        with torch.inference_mode():
            # Reference: native silu(gate) * up MoE on the [gate; up] layout.
            # Reuses the in-tree block-fp8 MoE reference; it derives its own
            # softmax/topk routing from ``score``.
            ref = torch_w8a8_block_fp8_moe(
                a, w13, w2, w13_s, w2_s, score, self.TOPK, BLOCK_SIZE
            )

            # Same routing for every cutlass path under test.
            topk_output = select_experts(
                hidden_states=a,
                router_logits=score,
                topk_config=TopKConfig(top_k=self.TOPK, renormalize=False),
            )
            topk_weights, topk_ids, _ = topk_output

            # Cutlass with the gate/up swap applied -> must compute silu(gate)*up.
            w13_swapped, w13_s_swapped = swap_w13_gate_up_halves(w13, w13_s)
            out_swapped = self._run_cutlass(
                a, w13_swapped, w2, w13_s_swapped, w2_s, topk_weights, topk_ids
            )

            # Cutlass WITHOUT the swap -> computes the wrong function.
            out_unswapped = self._run_cutlass(
                a, w13, w2, w13_s, w2_s, topk_weights, topk_ids
            )

        def rel_err(x, y):
            x, y = x.to(torch.float32), y.to(torch.float32)
            return (
                torch.mean(torch.abs(x - y)) / torch.mean(torch.abs(y))
            ).item()

        err_swapped = rel_err(out_swapped, ref)
        err_unswapped = rel_err(out_unswapped, ref)

        # The swapped path reproduces the reference within tolerance.
        self.assertLess(
            err_swapped,
            0.05,
            f"swapped cutlass MoE diverges from silu(gate)*up reference "
            f"(rel err {err_swapped:.4f})",
        )
        # The unswapped path computes a materially different function. This is
        # the operand-order regression the swap exists to prevent.
        self.assertGreater(
            err_unswapped,
            0.02,
            f"unswapped cutlass MoE unexpectedly matched the reference "
            f"(rel err {err_unswapped:.4f}); the gate/up swap may be a no-op",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
