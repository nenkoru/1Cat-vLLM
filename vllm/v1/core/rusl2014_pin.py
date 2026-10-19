# SPDX-License-Identifier: Apache-2.0
"""RUSL-2014: hot prefixes stay in the GPU prefix cache.

The GPU prefix cache evicts the least recently freed block first. A long cold request (33 thousand tokens of a new
neighbour) rotates the whole queue and takes with it the blocks every agent hits all the time: the system prompt,
the tool list, the GDN state at the end of them. This module keeps such blocks out of the eviction queue.

Score of a block hash: the number of prefix hits on it, halved every ``half_life_s`` seconds. A hit is a call of
``BlockPool.touch`` (a new request reuses the cached block), so a block shared by all sessions gathers score fast,
the history of one session gathers one hit per turn, and the blocks of a one-shot request gather none. The score
lives with the hash, not with the block: a hot prefix that was evicted anyway is hot again as soon as it is cached.

One hit is enough by default: the GDN state at the boundary a session reached is hit exactly once (by its next
turn), and without it the hit of a hybrid model falls back to an older state; blocks of one-shot requests get no hit
at all and are never pinned.

When a cached block is freed (``ref_cnt`` reaches 0) and its score is at least ``min_hits``, it is pinned instead of
being appended to the free queue, up to ``budget`` blocks; a hotter block displaces the coldest pinned one (which
goes back to the queue like any freed block). Pinned blocks:

- stay in the cache map, so lookups hit them; a hit takes the block out of the pinned set (it is in use again);
- count as free blocks (``BlockPool.get_num_free_blocks``): admission and preemption see the same capacity as
  without pins, so a pin never makes the scheduler preempt a running request;
- are evicted only when the queue alone cannot serve an allocation (``forced``, the coldest first) — this counter
  is the number of evictions of pinned prefixes;
- go back to the head of the queue (evicted first) after ``idle_s`` seconds without a hit.

Switch: ``VLLM_KV_PIN_HOT=1`` (default ``0``: no object is created, the block pool behaves as before).
  ``VLLM_KV_PIN_HOT_BLOCKS`` — budget, blocks or percent of the pool (default ``10%``);
  ``VLLM_KV_PIN_HOT_MIN_HITS`` — score needed to pin (default 1);
  ``VLLM_KV_PIN_HOT_HALF_LIFE_S`` — half-life of the score (default 900);
  ``VLLM_KV_PIN_HOT_IDLE_S`` — release a pinned block without hits for this long (default 1800);
  ``VLLM_KV_PIN_HOT_LOG_S`` — period of the counters line in the log (default 60).
"""

from __future__ import annotations

import heapq
import math
import os
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

logger = init_logger(__name__)

ENV = "VLLM_KV_PIN_HOT"
COUNTERS = ("pins", "displaced", "idle", "forced", "dropped", "hits_on_pinned")


def _parse_budget(value: str, num_gpu_blocks: int) -> int:
    value = value.strip()
    if value.endswith("%"):
        budget = int(num_gpu_blocks * float(value[:-1]) / 100)
    else:
        budget = int(value)
    if budget < 0:
        raise ValueError(f"{ENV}_BLOCKS={value!r}: must not be negative")
    return min(budget, num_gpu_blocks - 1)


class HotPins:
    """Pinned set of hot cached blocks of one BlockPool (scheduler process, single thread)."""

    def __init__(
        self,
        budget: int,
        min_hits: float = 1.0,
        half_life_s: float = 900.0,
        idle_s: float = 1800.0,
        log_s: float = 60.0,
        max_scores: int = 1 << 16,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if half_life_s <= 0:
            raise ValueError(f"{ENV}_HALF_LIFE_S must be positive")
        if min_hits <= 0:
            raise ValueError(f"{ENV}_MIN_HITS must be positive")
        self.budget = budget
        self.min_hits = min_hits
        self.half_life_s = half_life_s
        self.idle_s = idle_s
        self.log_s = log_s
        self.max_scores = max_scores
        self._clock = clock
        # block hash -> [score, time of the score]; LRU by the last hit
        self._scores: OrderedDict[Any, list[float]] = OrderedDict()
        # block_id -> (block, priority, time of the last hit)
        self.pinned: dict[int, tuple[KVCacheBlock, float, float]] = {}
        # min-heap of (priority, block_id); entries of blocks no longer pinned are skipped lazily
        self._heap: list[tuple[float, int]] = []
        self.counters = dict.fromkeys(COUNTERS, 0)
        self.max_pinned = 0
        self._next_sweep = 0.0
        self._next_log = 0.0
        self._logged: tuple[int, ...] | None = None

    @classmethod
    def from_env(cls, num_gpu_blocks: int) -> HotPins | None:
        if os.environ.get(ENV, "0").strip() not in ("1", "true", "on"):
            return None
        budget = _parse_budget(
            os.environ.get(f"{ENV}_BLOCKS", "10%"), num_gpu_blocks
        )
        pins = cls(
            budget=budget,
            min_hits=float(os.environ.get(f"{ENV}_MIN_HITS", "1")),
            half_life_s=float(os.environ.get(f"{ENV}_HALF_LIFE_S", "900")),
            idle_s=float(os.environ.get(f"{ENV}_IDLE_S", "1800")),
            log_s=float(os.environ.get(f"{ENV}_LOG_S", "60")),
            max_scores=max(1 << 12, 8 * num_gpu_blocks),
        )
        logger.info(
            "RUSL-2014: hot prefix pinning is ON (%s=1): budget %d of %d blocks, "
            "min hits %.1f, half-life %.0f s, idle release %.0f s",
            ENV,
            pins.budget,
            num_gpu_blocks,
            pins.min_hits,
            pins.half_life_s,
            pins.idle_s,
        )
        return pins

    # --- scores ---

    def _score(self, key: Any, now: float) -> float:
        entry = self._scores.get(key)
        if entry is None:
            return 0.0
        score, t = entry
        return score * 0.5 ** ((now - t) / self.half_life_s)

    def _priority(self, score: float, now: float) -> float:
        # log2 of the score brought to time 0: the order of two decaying scores never changes,
        # so a pinned block (whose score cannot change: a hit unpins it) keeps its heap position.
        return math.log2(score) + now / self.half_life_s

    def on_hit(self, block: KVCacheBlock) -> None:
        """A request reuses the cached block (``BlockPool.touch``)."""
        key = block.block_hash
        if key is None:
            return
        now = self._clock()
        entry = self._scores.get(key)
        if entry is None:
            self._scores[key] = [1.0, now]
            if len(self._scores) > self.max_scores:
                self._scores.popitem(last=False)
        else:
            entry[0] = entry[0] * 0.5 ** ((now - entry[1]) / self.half_life_s) + 1.0
            entry[1] = now
            self._scores.move_to_end(key)

    # --- pinned set ---

    def is_pinned(self, block: KVCacheBlock) -> bool:
        return block.block_id in self.pinned

    def take(self, block: KVCacheBlock) -> bool:
        """The pinned block is hit and goes back into use. False if it was not pinned."""
        if self.pinned.pop(block.block_id, None) is None:
            return False
        self.counters["hits_on_pinned"] += 1
        return True

    def drop(self, block: KVCacheBlock) -> bool:
        """The pinned block loses its hash (``BlockPool.evict_blocks``); the caller queues it."""
        if self.pinned.pop(block.block_id, None) is None:
            return False
        self.counters["dropped"] += 1
        return True

    def _peek_coldest(self) -> tuple[float, int] | None:
        """(priority, block_id) of the coldest pinned block; drops stale heap entries."""
        heap = self._heap
        while heap:
            priority, block_id = heap[0]
            item = self.pinned.get(block_id)
            if item is not None and item[1] == priority:
                return priority, block_id
            heapq.heappop(heap)
        return None

    def _pop_coldest(self) -> KVCacheBlock | None:
        top = self._peek_coldest()
        if top is None:
            return None
        heapq.heappop(self._heap)
        return self.pinned.pop(top[1])[0]

    def offer(
        self, blocks: Iterable[KVCacheBlock]
    ) -> tuple[list[KVCacheBlock], list[KVCacheBlock]]:
        """Freed cached blocks (``ref_cnt`` 0, hash set), in queue order.

        Returns (blocks for the free queue in the same order, displaced pinned blocks for the queue).
        """
        if self.budget <= 0:
            return list(blocks), []
        now = self._clock()
        queue: list[KVCacheBlock] = []
        displaced: list[KVCacheBlock] = []
        for block in blocks:
            score = self._score(block.block_hash, now)
            if score < self.min_hits:
                queue.append(block)
                continue
            priority = self._priority(score, now)
            if len(self.pinned) >= self.budget:
                coldest = self._peek_coldest()
                if coldest is None or coldest[0] >= priority:
                    # not hotter than anything pinned
                    queue.append(block)
                    continue
                displaced_block = self._pop_coldest()
                assert displaced_block is not None
                displaced.append(displaced_block)
                self.counters["displaced"] += 1
            entry = self._scores.get(block.block_hash)
            last_hit = entry[1] if entry is not None else now
            self.pinned[block.block_id] = (block, priority, last_hit)
            heapq.heappush(self._heap, (priority, block.block_id))
            self.counters["pins"] += 1
        self.max_pinned = max(self.max_pinned, len(self.pinned))
        if len(self._heap) > 4 * max(self.budget, 16):
            self._rebuild_heap()
        return queue, displaced

    def _rebuild_heap(self) -> None:
        self._heap = [(item[1], block_id) for block_id, item in self.pinned.items()]
        heapq.heapify(self._heap)

    def reclaim(self, num_blocks: int) -> list[KVCacheBlock]:
        """Unpin up to ``num_blocks`` coldest blocks to be evicted now (the queue is short)."""
        out: list[KVCacheBlock] = []
        while len(out) < num_blocks:
            block = self._pop_coldest()
            if block is None:
                break
            out.append(block)
        self.counters["forced"] += len(out)
        if out:
            logger.warning(
                "RUSL-2014: %d pinned prefix blocks evicted: the free queue could not "
                "serve an allocation (budget %d is too large for this load)",
                len(out),
                self.budget,
            )
        return out

    def release_all(self) -> list[KVCacheBlock]:
        out = [item[0] for item in self.pinned.values()]
        self.pinned.clear()
        self._heap.clear()
        return out

    def sweep_idle(self) -> list[KVCacheBlock]:
        """Pinned blocks without hits for ``idle_s``; checked at most every 10 s."""
        now = self._clock()
        if now < self._next_sweep or not self.pinned:
            return []
        self._next_sweep = now + min(10.0, self.idle_s / 4)
        idle = [
            block_id
            for block_id, (_, _, last_hit) in self.pinned.items()
            if now - last_hit > self.idle_s
        ]
        out = [self.pinned.pop(block_id)[0] for block_id in idle]
        self.counters["idle"] += len(out)
        return out

    def maybe_log(self) -> None:
        now = self._clock()
        if now < self._next_log:
            return
        self._next_log = now + self.log_s
        state = (len(self.pinned), *self.counters.values())
        if state == self._logged:
            return
        self._logged = state
        logger.info(
            "RUSL-2014 pin-hot: pinned %d/%d (max %d), pins %d, displaced %d, "
            "idle %d, forced %d, dropped %d, hits on pinned %d",
            len(self.pinned),
            self.budget,
            self.max_pinned,
            *self.counters.values(),
        )
