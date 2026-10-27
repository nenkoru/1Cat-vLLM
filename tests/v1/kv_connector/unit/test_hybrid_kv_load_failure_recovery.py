# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV load failure handling with the hybrid memory allocator (several KV
cache groups, e.g. attention + Mamba)."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

BLOCK_SIZE = 16
NUM_COMPUTED_BLOCKS = 4

# req_id -> block ids per KV cache group. The Mamba group keeps its state in
# the last block of the computed prefix (earlier entries are the null block).
HYBRID_BLOCK_IDS = {
    "r1": ([1, 2, 3, 4], [0, 0, 0, 5]),
    "r2": ([1, 2, 6, 7], [0, 0, 0, 8]),
}
SINGLE_GROUP_BLOCK_IDS = {"r1": ([1, 2, 3, 4],)}


def _make_request(req_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=req_id, num_computed_tokens=NUM_COMPUTED_BLOCKS * BLOCK_SIZE
    )


def _make_scheduler(
    block_ids: dict[str, tuple[list[int], ...]],
    recompute: bool = True,
    running: list[SimpleNamespace] | None = None,
    waiting_for_remote_kvs: list[SimpleNamespace] | None = None,
) -> Scheduler:
    scheduler = object.__new__(Scheduler)
    scheduler.block_size = BLOCK_SIZE
    scheduler.recompute_kv_load_failures = recompute
    scheduler.kv_cache_manager = Mock()
    scheduler.kv_cache_manager.get_block_ids.side_effect = lambda r: block_ids[r]
    num_groups = len(next(iter(block_ids.values())))
    scheduler.kv_cache_config = SimpleNamespace(kv_cache_groups=[None] * num_groups)
    scheduler.running = running or []
    scheduler.skipped_waiting = []
    for request in waiting_for_remote_kvs or []:
        request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
        scheduler.skipped_waiting.append(request)
    scheduler.failed_recving_kv_req_ids = set()
    return scheduler


def _evicted(scheduler: Scheduler) -> set[int]:
    evicted: set[int] = set()
    for call in scheduler.kv_cache_manager.evict_blocks.call_args_list:
        evicted |= set(call.args[0])
    return evicted


@pytest.mark.parametrize("invalid_block_id", [2, 4, 5])
def test_async_load_failure_recomputes_from_scratch(invalid_block_id: int):
    """An invalid block in any group restarts the request from token 0."""
    scheduler = _make_scheduler(HYBRID_BLOCK_IDS)
    request = _make_request("r1")

    affected, num_tokens, blocks_to_evict = (
        scheduler._update_requests_with_invalid_blocks(
            [request], {invalid_block_id}, {}, evict_blocks=False
        )
    )

    assert affected == {"r1"}
    assert request.num_computed_tokens == 0
    assert num_tokens == NUM_COMPUTED_BLOCKS * BLOCK_SIZE
    assert not blocks_to_evict


def test_unaffected_request_is_unchanged():
    scheduler = _make_scheduler(HYBRID_BLOCK_IDS)
    request = _make_request("r1")

    affected, num_tokens, blocks_to_evict = (
        scheduler._update_requests_with_invalid_blocks(
            [request], {99}, {}, evict_blocks=True
        )
    )

    assert not affected and num_tokens == 0 and not blocks_to_evict
    assert request.num_computed_tokens == NUM_COMPUTED_BLOCKS * BLOCK_SIZE


def test_invalid_block_beyond_computed_prefix_is_ignored():
    scheduler = _make_scheduler(HYBRID_BLOCK_IDS)
    request = _make_request("r1")
    # Only the first block holds computed tokens; the rest is being scheduled.
    request.num_computed_tokens = BLOCK_SIZE

    affected, _, _ = scheduler._update_requests_with_invalid_blocks(
        [request], {3}, {}, evict_blocks=True
    )

    assert not affected
    assert request.num_computed_tokens == BLOCK_SIZE


def test_sync_load_failure_collects_blocks_to_evict():
    scheduler = _make_scheduler(HYBRID_BLOCK_IDS)
    requests = [_make_request("r1"), _make_request("r2")]

    affected, _, blocks_to_evict = scheduler._update_requests_with_invalid_blocks(
        requests, {2}, {}, evict_blocks=True
    )

    # Block 2 is shared by both requests: both are affected.
    assert affected == {"r1", "r2"}
    assert all(r.num_computed_tokens == 0 for r in requests)
    # The invalid block and the downstream blocks of its group.
    assert blocks_to_evict == {2, 3, 4, 6, 7}


@pytest.mark.parametrize("recompute", [True, False])
def test_handle_invalid_blocks_hybrid(recompute: bool):
    running = _make_request("r1")
    waiting = _make_request("r2")
    scheduler = _make_scheduler(
        HYBRID_BLOCK_IDS,
        recompute=recompute,
        running=[running],
        waiting_for_remote_kvs=[waiting],
    )

    failed = scheduler._handle_invalid_blocks({2, 8}, {})

    if recompute:
        # The running request cannot be rescheduled and is returned to be
        # failed by update_from_output; the async one is retried.
        assert failed == {"r1"}
        assert scheduler.failed_recving_kv_req_ids == {"r2"}
        assert waiting.num_computed_tokens == 0
    else:
        assert failed == {"r1", "r2"}
        assert not scheduler.failed_recving_kv_req_ids
    # The failed running request will not recompute its blocks, so they must
    # leave the prefix cache under any policy.
    assert _evicted(scheduler) == {2, 3, 4}


def test_handle_invalid_blocks_single_group_unchanged():
    """Without the hybrid memory allocator the recompute policy still
    reschedules running requests from the longest valid prefix."""
    running = _make_request("r1")
    scheduler = _make_scheduler(SINGLE_GROUP_BLOCK_IDS, running=[running])

    failed = scheduler._handle_invalid_blocks({3}, {})

    assert failed == {"r1"}
    assert running.num_computed_tokens == 2 * BLOCK_SIZE
    scheduler.kv_cache_manager.evict_blocks.assert_not_called()
