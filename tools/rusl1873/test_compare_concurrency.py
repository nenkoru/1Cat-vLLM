# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import tempfile
import unittest

from compare_concurrency import compare


def arm(candidate=False):
    result = {
        "manifest": {"seats": 24 if candidate else 16, "qpn2": False,
                     "wheel_sha256": "test-wheel", "model_revision": "test-model",
                     "gpu": "test-gpu", "server_argv": "test-24" if candidate else "test-16"},
        "settings": {"contexts": [8000,33000], "clients": [16,24,32], "rounds": 3, "tokens": 128},
        "stages": [],
    }
    for context in [8000,33000]:
        for clients in [16,24,32]:
            result["stages"].append({
                "context": context, "clients": clients, "prompt_sha256": f"{context}-{clients}",
                "warmup": {"valid": True, "preemptions": 0},
                "rounds": [{"valid": True, "preemptions": 0,
                            "completion_tok_s": 110 if candidate else 100,
                            "rows": [{"turn_s": 9 if candidate else 10} for _ in range(clients)]}
                           for _ in range(3)],
            })
    return result


class CompareTest(unittest.TestCase):
    def check(self, arms):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for index, data in enumerate(arms):
                path = Path(tmp) / f"{index}.json"
                path.write_text(json.dumps(data))
                paths.append(path)
            return compare(paths)

    def test_full_matrix_passes_only_performance(self):
        result = self.check([arm(),arm(True),arm(True),arm()])
        self.assertTrue(result["performance_gate"])
        self.assertFalse(result["rollout_approved"])
        self.assertIn("not evaluated", result["quality_gate"])

    def test_partial_matrix_is_not_certified(self):
        arms = [arm(),arm(True),arm(True),arm()]
        arms[2]["stages"].pop()
        with self.assertRaisesRegex(ValueError, "missing"):
            self.check(arms)

    def test_prompt_or_model_mismatch_rejected(self):
        for mismatch in ["prompt", "model"]:
            arms = [arm(),arm(True),arm(True),arm()]
            if mismatch == "prompt":
                arms[1]["stages"][0]["prompt_sha256"] = "different"
            else:
                arms[1]["manifest"]["model_revision"] = "different"
            with self.assertRaises(ValueError):
                self.check(arms)

    def test_latency_regression_and_preemption_fail(self):
        for failure in ["latency", "preemption"]:
            arms = [arm(),arm(True),arm(True),arm()]
            for index in [1,2]:
                for sample in arms[index]["stages"][0]["rounds"]:
                    if failure == "latency":
                        for row in sample["rows"]:
                            row["turn_s"] = 12
                    else:
                        sample["preemptions"] = 1
            self.assertFalse(self.check(arms)["performance_gate"])


if __name__ == "__main__":
    unittest.main()
