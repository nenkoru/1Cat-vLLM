# SPDX-License-Identifier: Apache-2.0
import json
import datetime
import itertools
from pathlib import Path
import tempfile
import unittest

from compare_concurrency import compare
from concurrency import COUNTERS, summarize

RUNS = itertools.count()


def sample(context, clients, candidate, tokens):
    rows = [{"turn_s": (9 if candidate else 10) if tokens == 128 else .05,
             "first_s": .01, "completion_tokens": tokens,
             "prompt_tokens": context, "output_sha256": "test-output"}
            for _ in range(clients)]
    elapsed = tokens * clients / (110 if candidate else 100)
    metrics = {COUNTERS[0]: 0}
    return {**summarize(rows, elapsed, tokens, metrics, metrics), "rows": rows,
            "elapsed_s": elapsed, "metrics_before": metrics, "metrics_after": metrics}


def arm(candidate=False):
    run = next(RUNS)
    result = {
        "manifest": {"seats": 24 if candidate else 16, "qpn2": False,
                     "arm": f"run-{run}",
                     "started_msk": datetime.datetime.fromtimestamp(run, datetime.timezone.utc).isoformat(),
                     "wheel_sha256": "test-wheel", "model_revision": "test-model",
                     "gpu": "test-gpu", "server_argv": "test-24" if candidate else "test-16"},
        "settings": {"contexts": [8000,33000], "clients": [16,24,32], "rounds": 3, "tokens": 128},
        "stages": [],
    }
    for context in [8000,33000]:
        for clients in [16,24,32]:
            result["stages"].append({
                "context": context, "clients": clients, "prompt_sha256": f"{context}-{clients}",
                "warmup": sample(context, clients, candidate, 1),
                "rounds": [sample(context, clients, candidate, 128) for _ in range(3)],
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
                        sample.update(summarize(sample["rows"], sample["elapsed_s"], 128,
                                                sample["metrics_before"], sample["metrics_after"]))
                    else:
                        sample["preemptions"] = 1
            if failure == "latency":
                self.assertFalse(self.check(arms)["performance_gate"])
            else:
                with self.assertRaises(ValueError):
                    self.check(arms)

    def test_review_invalid_workload_rows_usage_and_counters(self):
        for failure in ["tokens", "rows", "usage", "cancelled_counts", "nan", "error"]:
            with self.subTest(failure=failure):
                arms = [arm(), arm(True), arm(True), arm()]
                for data in arms:
                    if failure == "tokens":
                        data["settings"]["tokens"] = 1
                    stage = data["stages"][0]
                    for index, measured in enumerate(stage["rounds"]):
                        if failure == "rows":
                            measured["rows"] = measured["rows"][:1]
                        elif failure == "usage":
                            measured["rows"][0]["completion_tokens"] = 1
                        elif failure == "cancelled_counts":
                            measured["preemptions"] = [-1, 1, 0][index]
                        elif failure == "nan":
                            measured["rows"][0]["turn_s"] = float('nan')
                        elif failure == "error":
                            measured["rows"][0]["error"] = 'HTTP 500'
                with self.assertRaises(ValueError):
                    self.check(arms)

    def test_review_unknown_identity_is_not_equality(self):
        for field in ["wheel_sha256", "model_revision", "gpu", "server_argv", "prompt_sha256"]:
            arms = [arm(), arm(True), arm(True), arm()]
            for data in arms:
                if field == "prompt_sha256":
                    data["stages"][0][field] = None
                else:
                    data["manifest"][field] = None
            with self.assertRaisesRegex(ValueError, "identity"):
                self.check(arms)

    def test_review_reused_file_or_run_is_not_independent(self):
        a, b = arm(), arm(True)
        with self.assertRaisesRegex(ValueError, "distinct runs"):
            self.check([a, b, b, a])
        with tempfile.TemporaryDirectory() as tmp:
            p, q = Path(tmp)/'a.json', Path(tmp)/'b.json'
            p.write_text(json.dumps(a)); q.write_text(json.dumps(b))
            with self.assertRaisesRegex(ValueError, "duplicate arm file"):
                compare([p, q, q, p])


if __name__ == "__main__":
    unittest.main()
