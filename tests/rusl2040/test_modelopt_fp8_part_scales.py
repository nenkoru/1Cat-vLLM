# SPDX-License-Identifier: Apache-2.0
"""RUSL-2040: VLLM_SM70_MODELOPT_FP8_PART_SCALES keeps the per-partition
checkpoint scales of the fused ModelOpt FP8 projections on TurboMind.

Without the switch the preparation is the release one (requantization to the
largest scale). Runs on CPU: the native operators are replaced by stand-ins.
"""

import pytest
import torch

from vllm import _sm70_ops
from vllm.model_executor.layers.quantization import modelopt
from vllm.model_executor.layers.quantization import (
    sm70_modelopt_fp8_part_scales as part_scales,
)

FP8 = torch.float8_e4m3fn
ENV = "VLLM_SM70_MODELOPT_FP8_PART_SCALES"


def _layer(prefix, k, widths, scales):
    layer = torch.nn.Module()
    layer.prefix = prefix
    layer.input_size_per_partition = k
    layer.output_size_per_partition = sum(widths)
    layer.logical_widths = list(widths)
    weight = (torch.randn(sum(widths), k) * 0.1).to(FP8)
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(
        torch.tensor(scales, dtype=torch.float32), requires_grad=False
    )
    layer.input_scale = torch.nn.Parameter(
        torch.ones(len(widths)), requires_grad=False
    )
    return layer


def _qkvz(scales=(0.002, 0.002, 0.002, 0.003)):
    return _layer(
        "model.layers.0.linear_attn.in_proj_qkvz", 5120, [1024, 1024, 3072, 3072], scales
    )


def _method(turbomind=True):
    method = object.__new__(modelopt.ModelOptFp8LinearMethod)
    method.use_sm70_fp8_turbomind = turbomind
    method.input_dtype = torch.float16
    return method


class _Native:
    def __init__(self):
        self.calls = []

    def fp8_sm70_prepare(self, weight, block_scales, group_size):
        self.calls.append(("tm_prepare", weight.clone(), block_scales.clone()))
        n, k = weight.shape
        meta = torch.tensor([k, n])
        return torch.zeros(k, n, dtype=torch.uint8), block_scales.half(), meta

    def fp8_gemm_sm70_out(self, out, x, *args):
        self.calls.append(("tm", x.shape[0]))


@pytest.fixture
def native(monkeypatch):
    stand = _Native()
    for name in ("fp8_sm70_prepare", "fp8_gemm_sm70_out"):
        monkeypatch.setattr(_sm70_ops, name, getattr(stand, name))
    monkeypatch.setattr(torch.ops._C, "fp8_sm70_prepare", object(), raising=False)
    requant = []

    def fake_requant(weight, scales, widths):
        requant.append(list(widths))
        return scales.max(), weight

    monkeypatch.setattr(modelopt, "requantize_with_max_scale", fake_requant)
    monkeypatch.delenv(ENV, raising=False)
    return monkeypatch, stand, requant


@pytest.mark.parametrize("value, on", [(None, False), ("0", False), ("1", True)])
def test_switch_value(monkeypatch, value, on):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    assert part_scales.enabled() is on


def test_row_and_block_scales_keep_each_partition():
    rows = part_scales.row_scales(_qkvz())
    assert rows.shape == (8192,)
    assert torch.all(rows[:5120] == torch.tensor(0.002))
    assert torch.all(rows[5120:] == torch.tensor(0.003))
    blocks = part_scales.block_scales(rows, 5120)
    assert blocks.shape == (64, 40)
    assert torch.all(blocks[:40] == torch.tensor(0.002))
    assert torch.all(blocks[40:] == torch.tensor(0.003))


def test_qkv_proj_partitions():
    qkv = _layer("model.layers.3.self_attn.qkv_proj", 5120, [6144, 512, 512],
                 [0.0007, 0.0005, 0.0006])
    blocks = part_scales.block_scales(part_scales.row_scales(qkv), 5120)
    assert blocks.shape == (56, 40)
    assert torch.all(blocks[:48] == torch.tensor(0.0007))
    assert torch.all(blocks[48:52] == torch.tensor(0.0005))
    assert torch.all(blocks[52:] == torch.tensor(0.0006))


def test_row_scales_refuse_unloaded_or_unaligned():
    unloaded = _qkvz(scales=(0.002, torch.finfo(torch.float32).min, 0.002, 0.003))
    assert part_scales.row_scales(unloaded) is None
    odd = _layer("x.qkv_proj", 5120, [6144, 448, 576], [0.1, 0.2, 0.3])
    assert part_scales.row_scales(odd) is None
    fused = _qkvz()
    fused.weight_scale = torch.nn.Parameter(torch.tensor([0.003]), requires_grad=False)
    assert part_scales.row_scales(fused) is None


def test_switch_off_keeps_release_preparation(native):
    _, stand, requant = native
    layer = _qkvz()
    _method().process_weights_after_loading(layer)
    assert requant == [[1024, 1024, 3072, 3072]]
    (tm,) = stand.calls
    assert tm[0] == "tm_prepare"
    assert torch.all(tm[2] == torch.tensor(0.003))  # one max scale everywhere
    assert layer.sm70_modelopt_fp8_turbomind


def test_switch_on_keeps_checkpoint_codes_and_scales(native):
    monkeypatch, stand, requant = native
    monkeypatch.setenv(ENV, "1")
    layer = _qkvz()
    original = layer.weight.detach().clone()
    _method().process_weights_after_loading(layer)
    assert requant == []
    (tm,) = stand.calls
    _, t_weight, t_blocks = tm
    assert torch.equal(t_weight.view(torch.uint8), original.view(torch.uint8))
    assert torch.all(t_blocks[:40] == torch.tensor(0.002))
    assert torch.all(t_blocks[40:] == torch.tensor(0.003))
    assert layer.sm70_modelopt_fp8_turbomind
    assert layer.sm70_modelopt_fp8_group_size == 128


def test_switch_on_unaligned_layer_is_requantized(native):
    monkeypatch, stand, requant = native
    monkeypatch.setenv(ENV, "1")
    layer = _layer("x.qkv_proj", 5120, [6144, 448, 576], [0.1, 0.2, 0.3])
    _method().process_weights_after_loading(layer)
    assert requant == [[6144, 448, 576]]
    (tm,) = stand.calls
    assert torch.all(tm[2] == torch.tensor(0.3))


def test_switch_on_single_partition_is_unchanged(native):
    monkeypatch, stand, requant = native
    monkeypatch.setenv(ENV, "1")
    out = _layer("model.layers.0.linear_attn.out_proj", 3072, [5120], [0.01])
    _method().process_weights_after_loading(out)
    assert requant == []
    (tm,) = stand.calls
    assert tm[2].shape == (40, 24)
    assert torch.all(tm[2] == torch.tensor(0.01))


def test_apply_is_the_release_call(native):
    monkeypatch, stand, _ = native
    monkeypatch.setenv(ENV, "1")
    layer = _qkvz()
    _method().process_weights_after_loading(layer)
    stand.calls.clear()
    y = _method().apply(layer, torch.zeros(2, 3, 5120, dtype=torch.float16))
    assert y.shape == (2, 3, 8192)
    assert stand.calls == [("tm", 6)]
