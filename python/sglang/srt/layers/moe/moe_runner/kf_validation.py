from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Optional

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.utils import get_bool_env_var, get_device_capability

_QWEN35_V2A_ENV = "SGLANG_KF_QWEN35_FP8_MOE_V2A"
_QWEN35_V2A_SRC_ENV = "SGLANG_KF_QWEN35_FP8_MOE_V2A_SOURCE_DIR"
_QWEN35_V2A_ALLOW_NON_H200_ENV = "SGLANG_KF_QWEN35_FP8_MOE_V2A_ALLOW_NON_H200"


def _candidate_source_dir() -> Path:
    env_dir = os.getenv(_QWEN35_V2A_SRC_ENV)
    if not env_dir:
        raise RuntimeError(f"{_QWEN35_V2A_SRC_ENV} must point at extracted 6e396 source")
    return Path(env_dir).expanduser().resolve()


def _cutlass_include_paths() -> list[str]:
    roots = []
    for env_name in ("CUTLASS_INCLUDE_DIR", "CUTLASS_INCLUDE_PATH"):
        env_value = os.getenv(env_name)
        if env_value:
            roots.extend(Path(part) for part in env_value.split(os.pathsep) if part)

    roots.extend(
        Path(path)
        for path in (
            "/usr/local/include",
            "/usr/include",
            "/opt/cutlass/include",
            "/workspace/cutlass/include",
            "/workspace/third_party/cutlass/include",
            "/workspace/sgl-kernel/3rdparty/cutlass/include",
            "/workspace/sgl-kernel/third_party/cutlass/include",
        )
    )
    required_headers = (
        Path("cutlass") / "cutlass.h",
        Path("cutlass") / "util" / "packed_stride.hpp",
    )
    return [
        str(root)
        for root in roots
        if any((root / header).exists() for header in required_headers)
    ]


@functools.lru_cache(maxsize=1)
def _load_qwen35_v2a_extension():
    from torch.utils.cpp_extension import load

    src_dir = _candidate_source_dir()
    sources = [src_dir / "main.cpp", src_dir / "kernel.cu"]
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"KF Qwen3.5 6e396 source files not found: {', '.join(missing)}"
        )

    return load(
        name=os.getenv(
            "KF_TORCH_EXTENSION_NAME", "sglang_kf_qwen35_fp8_moe_candidate"
        ),
        sources=[str(path) for path in sources],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-lineinfo",
            "-gencode=arch=compute_90a,code=sm_90a",
        ],
        extra_include_paths=_cutlass_include_paths(),
        extra_ldflags=["-lcuda"],
        verbose=get_bool_env_var("SGLANG_KF_QWEN35_FP8_MOE_V2A_VERBOSE", "false"),
    )


def _is_target_h200() -> bool:
    if get_bool_env_var(_QWEN35_V2A_ALLOW_NON_H200_ENV, "false"):
        return True
    if not torch.cuda.is_available():
        return False
    device_name = torch.cuda.get_device_name(torch.cuda.current_device()).lower()
    return "h200" in device_name


def _block_shape_is_target(block_shape: Optional[list[int]]) -> bool:
    return block_shape == [128, 128] or block_shape == (128, 128)


def _can_run_qwen35_v2a(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    runner_config: MoeRunnerConfig,
    *,
    b1: Optional[torch.Tensor],
    b2: Optional[torch.Tensor],
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    w1_scale: Optional[torch.Tensor],
    w2_scale: Optional[torch.Tensor],
    w1_zp: Optional[torch.Tensor],
    w2_zp: Optional[torch.Tensor],
    a1_scale: Optional[torch.Tensor],
    a2_scale: Optional[torch.Tensor],
    block_shape: Optional[list[int]],
) -> bool:
    if not get_bool_env_var(_QWEN35_V2A_ENV, "false"):
        return False
    if not hidden_states.is_cuda or not _is_target_h200():
        return False
    if get_device_capability(hidden_states.device.index or 0) != (9, 0):
        return False
    if runner_config.num_local_experts != 64:
        return False
    if runner_config.hidden_size != 4096:
        return False
    if runner_config.intermediate_size_per_partition != 1024:
        return False
    if runner_config.top_k != 10:
        return False
    if runner_config.activation != "silu" or not runner_config.is_gated:
        return False
    if runner_config.apply_router_weight_on_input or runner_config.no_combine:
        return False
    if runner_config.routed_scaling_factor not in (None, 1.0):
        return False
    if (
        runner_config.gemm1_alpha is not None
        or runner_config.gemm1_clamp_limit is not None
    ):
        return False
    if runner_config.swiglu_limit is not None:
        return False
    if b1 is not None or b2 is not None:
        return False
    if not use_fp8_w8a8 or use_int8_w8a8 or use_int8_w8a16 or use_int4_w4a16:
        return False
    if per_channel_quant:
        return False
    if (
        w1_zp is not None
        or w2_zp is not None
        or a1_scale is not None
        or a2_scale is not None
    ):
        return False
    if w1_scale is None or w2_scale is None:
        return False
    if not _block_shape_is_target(block_shape):
        return False
    if hidden_states.shape[1] != 4096 or hidden_states.dtype != torch.bfloat16:
        return False
    if w1.shape != (64, 2048, 4096) or w2.shape != (64, 4096, 1024):
        return False
    if w1.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        return False
    if w1_scale.shape != (64, 16, 32) or w2_scale.shape != (64, 32, 8):
        return False
    if w1_scale.dtype != torch.float32 or w2_scale.dtype != torch.float32:
        return False
    if topk_ids.shape != topk_weights.shape or topk_ids.dim() != 2:
        return False
    if topk_ids.shape[0] != hidden_states.shape[0] or topk_ids.shape[1] != 10:
        return False
    if topk_ids.dtype != torch.int32 or topk_weights.dtype != torch.float32:
        return False
    tensors = (hidden_states, w1, w2, w1_scale, w2_scale, topk_ids, topk_weights)
    return all(tensor.is_contiguous() for tensor in tensors)


def maybe_run_qwen35_v2a(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    runner_config: MoeRunnerConfig,
    *,
    b1: Optional[torch.Tensor],
    b2: Optional[torch.Tensor],
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    w1_scale: Optional[torch.Tensor],
    w2_scale: Optional[torch.Tensor],
    w1_zp: Optional[torch.Tensor],
    w2_zp: Optional[torch.Tensor],
    a1_scale: Optional[torch.Tensor],
    a2_scale: Optional[torch.Tensor],
    block_shape: Optional[list[int]],
) -> Optional[torch.Tensor]:
    if not _can_run_qwen35_v2a(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        runner_config,
        b1=b1,
        b2=b2,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_zp=w1_zp,
        w2_zp=w2_zp,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
    ):
        return None

    out = torch.empty_like(hidden_states)
    module = _load_qwen35_v2a_extension()
    module.run(hidden_states, w1, w2, w1_scale, w2_scale, topk_ids, topk_weights, out)
    return out
