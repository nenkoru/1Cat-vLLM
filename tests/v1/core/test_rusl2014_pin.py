# SPDX-License-Identifier: Apache-2.0
"""RUSL-2014: hot prefixes pinned in the GPU prefix cache (VLLM_KV_PIN_HOT)."""

import pytest

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
from vllm.v1.core.rusl2014_pin import HotPins, _parse_budget


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make_pool(monkeypatch, n=12, budget=2, min_hits=2.0, idle_s=600.0):
    monkeypatch.setenv("VLLM_KV_PIN_HOT", "1")
    pool = BlockPool(num_gpu_blocks=n, enable_caching=True, hash_block_size=16)
    clock = Clock()
    pool._pins = HotPins(budget=budget, min_hits=min_hits, half_life_s=900.0, idle_s=idle_s, clock=clock)
    return pool, clock


def cache(pool, blocks, tag):
    """What cache_full_blocks does: give every block a hash and put it in the map."""
    for i, block in enumerate(blocks):
        key = make_block_hash_with_group_id(f"{tag}{i}".encode(), 0)
        block.block_hash = key
        pool.cached_block_hash_to_block.insert(key, block)


def hit(pool, block, times=1):
    """A later request reuses the cached block, then finishes."""
    for _ in range(times):
        pool.touch([block])
        pool.free_blocks([block])


def cached(pool, block):
    return pool.cached_block_hash_to_block.get_one_block(block.block_hash) is block


def test_switch_off_keeps_the_pool(monkeypatch):
    monkeypatch.delenv("VLLM_KV_PIN_HOT", raising=False)
    pool = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=16)
    assert pool._pins is None
    blocks = pool.get_new_blocks(3)
    cache(pool, blocks, "a")
    pool.free_blocks(blocks)
    assert pool.get_num_free_blocks() == pool.free_block_queue.num_free_blocks == 7
    pool.reclaim_pinned(100)  # no-op without the switch
    assert pool.get_num_free_blocks() == 7


def test_switch_on_from_env(monkeypatch):
    monkeypatch.setenv("VLLM_KV_PIN_HOT", "1")
    monkeypatch.setenv("VLLM_KV_PIN_HOT_BLOCKS", "25%")
    pool = BlockPool(num_gpu_blocks=100, enable_caching=True, hash_block_size=16)
    assert pool._pins is not None and pool._pins.budget == 25 and pool._pins.min_hits == 1.0
    # no prefix caching, nothing to pin
    assert BlockPool(num_gpu_blocks=100, enable_caching=False, hash_block_size=16)._pins is None
    assert _parse_budget("7", 100) == 7 and _parse_budget("200%", 10) == 9
    with pytest.raises(ValueError):
        _parse_budget("-1", 10)


def test_hot_block_survives_a_cold_rotation(monkeypatch):
    pool, _ = make_pool(monkeypatch, n=12, budget=2)
    hot = pool.get_new_blocks(1)[0]
    cache(pool, [hot], "sys")
    pool.free_blocks([hot])
    hit(pool, hot, times=2)  # score 2 -> pinned when freed
    assert pool._pins.is_pinned(hot)
    assert pool.get_num_free_blocks() == 11 and pool.free_block_queue.num_free_blocks == 10
    # a cold neighbour takes every queued block: the pinned one stays cached
    cold = pool.get_new_blocks(10)
    assert hot not in cold and cached(pool, hot)
    cache(pool, cold, "cold")
    pool.free_blocks(cold)
    assert not any(pool._pins.is_pinned(b) for b in cold), "blocks without hits are not pinned"
    again = pool.get_new_blocks(10)
    assert hot not in again and cached(pool, hot)
    assert pool._pins.counters["forced"] == 0
    # the next hit takes it back into use without touching the queue
    pool.touch([hot])
    assert hot.ref_cnt == 1 and not pool._pins.is_pinned(hot)
    assert pool._pins.counters["hits_on_pinned"] == 1


def test_pin_yields_to_an_allocation_instead_of_preemption(monkeypatch):
    pool, _ = make_pool(monkeypatch, n=6, budget=2)
    hot = pool.get_new_blocks(2)
    cache(pool, hot, "sys")
    pool.free_blocks(hot)
    for b in hot:
        hit(pool, b, times=3)
    hit(pool, hot[1], times=2)  # hot[1] is hotter, hot[0] goes first
    assert all(pool._pins.is_pinned(b) for b in hot)
    assert pool.get_num_free_blocks() == 5  # the scheduler sees the whole pool
    got = pool.get_new_blocks(4)
    assert hot[0] in got and hot[1] not in got
    assert not cached(pool, hot[0]) and cached(pool, hot[1])
    assert pool._pins.counters["forced"] == 1
    with pytest.raises(ValueError):
        pool.get_new_blocks(2)


def test_budget_keeps_the_hottest(monkeypatch):
    pool, clock = make_pool(monkeypatch, n=10, budget=2)
    blocks = pool.get_new_blocks(4)
    cache(pool, blocks, "b")
    pool.free_blocks(blocks)
    hit(pool, blocks[0], times=2)
    hit(pool, blocks[1], times=3)
    assert pool._pins.is_pinned(blocks[0]) and pool._pins.is_pinned(blocks[1])
    clock.t += 1
    hit(pool, blocks[2], times=5)  # hotter than blocks[0]: displaces it
    assert pool._pins.is_pinned(blocks[2]) and not pool._pins.is_pinned(blocks[0])
    assert pool._pins.counters["displaced"] == 1
    assert cached(pool, blocks[0]), "a displaced block is only queued, not evicted"
    assert len(pool._pins.pinned) == 2 and pool._pins.max_pinned == 2
    # a block colder than everything pinned is queued
    hit(pool, blocks[3], times=2)
    assert not pool._pins.is_pinned(blocks[3])
    assert pool._pins.counters["displaced"] == 1
    # free count stays consistent: 9 usable blocks, none in use
    assert pool.get_num_free_blocks() == 9


def test_score_decays(monkeypatch):
    pool, clock = make_pool(monkeypatch, n=8, budget=4, min_hits=2.0)
    b = pool.get_new_blocks(1)[0]
    cache(pool, [b], "d")
    pool.free_blocks([b])
    hit(pool, b, times=1)
    clock.t += 900  # one half-life: score 0.5, the next hit makes 1.5 < 2
    hit(pool, b, times=1)
    assert not pool._pins.is_pinned(b)
    hit(pool, b, times=1)  # 2.5
    assert pool._pins.is_pinned(b)


def test_idle_release(monkeypatch):
    pool, clock = make_pool(monkeypatch, n=8, budget=2, idle_s=600.0)
    b, other = pool.get_new_blocks(2)
    cache(pool, [b, other], "i")
    pool.free_blocks([b, other])
    hit(pool, b, times=2)
    assert pool._pins.is_pinned(b)
    clock.t += 601
    pool.touch([other])
    pool.free_blocks([other])  # any free sweeps
    assert not pool._pins.is_pinned(b) and cached(pool, b)
    assert pool._pins.counters["idle"] == 1
    # released to the queue head: evicted first
    assert pool.free_block_queue.fake_free_list_head.next_free_block is b


def test_evict_blocks_and_reset(monkeypatch):
    pool, _ = make_pool(monkeypatch, n=8, budget=2)
    a, b = pool.get_new_blocks(2)
    cache(pool, [a, b], "e")
    pool.free_blocks([a, b])
    hit(pool, a, times=2)
    hit(pool, b, times=2)
    pool.evict_blocks({a.block_id})
    assert not pool._pins.is_pinned(a) and a.block_hash is None
    assert pool._pins.counters["dropped"] == 1
    assert pool.get_num_free_blocks() == 7 and pool.free_block_queue.num_free_blocks == 6
    assert pool.reset_prefix_cache()
    assert not pool._pins.pinned and b.block_hash is None
    assert pool.free_block_queue.num_free_blocks == 7


def test_reclaim_before_store_on_evict_walk(monkeypatch):
    # the store-on-evict hook walks the queue head for num_blocks before the allocation
    pool, _ = make_pool(monkeypatch, n=5, budget=2)
    hot = pool.get_new_blocks(2)
    cache(pool, hot, "w")
    pool.free_blocks(hot)
    for b in hot:
        hit(pool, b, times=2)
    pool.reclaim_pinned(3)  # queue holds 2, needs 3: one pinned block to the head
    assert pool.free_block_queue.num_free_blocks == 3
    head = pool.free_block_queue.fake_free_list_head.next_free_block
    assert head in hot and head.block_hash is not None
