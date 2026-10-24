# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RUSL-2040: per-partition scales for the fused ModelOpt FP8 projections.

Swift-1.5 (Qwen3.8-27B, ModelOpt MIXED_PRECISION) keeps the attention qkv and
GDN qkvz projections in FP8 with one FP32 scale per checkpoint tensor: fused
qkvz has four partitions (q, k, v from in_proj_qkv, z from in_proj_z), qkv_proj
three (q, k, v). The release route requantizes the fused weight to the largest
scale, so every other partition is rounded a second time. On the real weights
(RUSL-2040, 24.10.2026) that is 0.8-3.2% max and ~1.1% rms error against fp32,
against 0.03-0.07% max and ~0.04% rms with the checkpoint codes.

TurboMind takes 128x128 block scales anyway, and every partition here is a
multiple of 128 rows, so each row block gets the scale of its partition: the
checkpoint codes stay as they are, the kernel and the memory are the same
(measured within +-1.5% at M 1-2048).
"""

from __future__ import annotations

import torch

from vllm import envs


def enabled() -> bool:
    return bool(envs.VLLM_SM70_MODELOPT_FP8_PART_SCALES)


def row_scales(layer: torch.nn.Module, block: int = 128) -> torch.Tensor | None:
    """FP32 scale of every output row from the per-partition checkpoint scales.

    None when the partitions cannot keep their own scales: a scale that was
    never loaded (the checkpoint fused the partitions under one scale) or a
    partition that is not a whole number of TurboMind row blocks.
    """
    scales = layer.weight_scale.detach().float().reshape(-1)
    widths = [int(w) for w in getattr(layer, "logical_widths", [])]
    if not widths or len(widths) != scales.numel():
        return None
    if sum(widths) != int(layer.weight.shape[0]):
        return None
    if any(w % block for w in widths) or not bool(torch.all(scales > 0).item()):
        return None
    sizes = torch.tensor(widths, device=scales.device)
    return torch.repeat_interleave(scales, sizes)


def block_scales(rows: torch.Tensor, k: int, block: int = 128) -> torch.Tensor:
    """TurboMind 128x128 block scales [N/128, K/128] from per-row scales."""
    out_blocks = rows.numel() // block
    in_blocks = (k + block - 1) // block
    per_block = rows[::block].reshape(out_blocks, 1)
    return per_block.expand(out_blocks, in_blocks).contiguous()
