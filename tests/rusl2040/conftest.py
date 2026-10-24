# SPDX-License-Identifier: Apache-2.0
import pytest


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    # CPU-only unit tests: no distributed state or accelerator cache to clean up.
    return False
