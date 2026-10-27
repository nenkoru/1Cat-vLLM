# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the debug line reported when the attention KV prefix hit of an
align-mode hybrid model reaches further than the last cached recurrent state."""

import hashlib
import logging
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.core import kv_cache_coordinator
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)

# These tests allocate only block metadata; no device/distributed state exists.
pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

BLOCK_SIZE = 32


class _Logger:
    def __init__(self, debug: bool):
        self.debug_enabled = debug
        self.lines: list[tuple] = []

    def isEnabledFor(self, level: int) -> bool:
        return level != logging.DEBUG or self.debug_enabled

    def debug(self, msg: str, *args) -> None:
        self.lines.append(args)


@pytest.fixture(params=[True, False], ids=["debug", "info"])
def log(request, monkeypatch):
    stub = _Logger(request.param)
    monkeypatch.setattr(kv_cache_coordinator, "logger", stub)
    return stub


def _request(name, length, hash_size=8):
    return SimpleNamespace(
        request_id=name,
        num_tokens=length,
        num_preemptions=0,
        shared_prefix_boundary=0,
        skip_reading_prefix_cache=False,
        block_hashes=[
            hashlib.sha256(f"{name}:{i}".encode()).digest()
            for i in range(length // hash_size)
        ],
    )


def _cache_manager(mamba_cache_mode="align"):
    full = FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float16
    )
    mamba = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((1,),),
        dtypes=(torch.float16,),
        mamba_cache_mode=mamba_cache_mode,
    )
    config = KVCacheConfig(
        1000,
        [],
        [KVCacheGroupSpec(["full"], full), KVCacheGroupSpec(["mamba"], mamba)],
    )
    return KVCacheManager(
        config, 100000, 8, use_eagle=False, prefix_cache_retention_interval=0
    )


def _prefill(cm, request):
    for start in range(0, request.num_tokens, BLOCK_SIZE):
        end = min(start + BLOCK_SIZE, request.num_tokens)
        for manager in cm.coordinator.single_type_managers:
            manager.new_step_starts()
            manager.remove_skipped_blocks(request.request_id, start)
            manager.allocate_new_blocks(request.request_id, end, end)
        cm.coordinator.cache_blocks(request, end)
    cm.coordinator.free(request.request_id)


def _sibling(a, name, shared_blocks):
    """A request that shares its first ``shared_blocks`` blocks with ``a``."""
    b = _request(name, 60 * BLOCK_SIZE + 3)
    shared_hashes = shared_blocks * BLOCK_SIZE // 8
    b.block_hashes[:shared_hashes] = a.block_hashes[:shared_hashes]
    return b


def test_gap_is_reported_once_per_lookup(log):
    cm = _cache_manager()
    a = _request("a", 80 * BLOCK_SIZE + 3)
    b = _sibling(a, "b", 30)
    _prefill(cm, a)
    for _ in range(100):  # a waiting request repeats the lookup every step
        _, hit = cm.get_computed_blocks(b)
        assert hit == 0
        assert b.shared_prefix_boundary == 30 * BLOCK_SIZE
    if log.debug_enabled:
        kv = 30 * BLOCK_SIZE
        assert log.lines == [(kv, 0, kv, b.num_tokens - 1)]
    else:
        assert log.lines == []


def test_no_gap_no_line(log):
    cm = _cache_manager()
    a = _request("a", 80 * BLOCK_SIZE + 3)
    _prefill(cm, a)
    _, hit = cm.get_computed_blocks(a)  # identical resend: state at the end
    assert hit == 80 * BLOCK_SIZE
    assert cm.coordinator.shared_prefix_boundary == 0
    assert log.lines == []


def test_other_prompts_are_reported_separately(log):
    cm = _cache_manager()
    a = _request("a", 80 * BLOCK_SIZE + 3)
    b, c = _sibling(a, "b", 30), _sibling(a, "c", 20)
    _prefill(cm, a)
    cm.get_computed_blocks(b)
    cm.get_computed_blocks(c)
    cm.get_computed_blocks(b)
    assert len(log.lines) == (2 if log.debug_enabled else 0)


def test_seen_gaps_are_bounded(monkeypatch):
    log = _Logger(debug=True)
    monkeypatch.setattr(kv_cache_coordinator, "logger", log)
    coordinator = _cache_manager().coordinator
    for i in range(3000):
        coordinator._log_state_gap([i.to_bytes(4, "big")], 100, 64, 32)
    assert len(log.lines) == 3000
    assert len(coordinator._logged_state_gaps) <= 1024
