# SPDX-License-Identifier: Apache-2.0
"""Evaluate complete A/B/B/A arm files; never certify partial observations."""
import argparse
import datetime
import json
import math
from pathlib import Path
import statistics

from concurrency import COUNTERS, percentile, summarize


def present(value):
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return bool(value) and all(present(k) and (positive(v) or present(v)) for k, v in value.items())
    return False


def positive(value):
    return type(value) in (float, int) and math.isfinite(value) and value > 0


def checked_round(sample, context, clients, tokens):
    if sample.get("valid") is not True or sample.get("errors") != []:
        raise ValueError("invalid measurement round")
    rows = sample.get("rows", [])
    if len(rows) != clients or not positive(sample.get("elapsed_s")):
        raise ValueError("incomplete rows or invalid elapsed time")
    for row in rows:
        if (row.get("error") or type(row.get("completion_tokens")) is not int
                or row["completion_tokens"] != tokens
                or type(row.get("prompt_tokens")) is not int
                or row["prompt_tokens"] != context
                or not positive(row.get("turn_s")) or not positive(row.get("first_s"))
                or not row["first_s"] <= row["turn_s"] <= sample["elapsed_s"]
                or not present(row.get("output_sha256"))):
            raise ValueError("invalid request usage, timing, error or identity")
    if sample.get("preemptions") != 0 or type(sample.get("preemptions")) is bool:
        raise ValueError("each round must have zero preemptions")
    before, after = sample.get("metrics_before", {}), sample.get("metrics_after", {})
    for metrics in [before, after]:
        count = metrics.get(COUNTERS[0])
        if type(count) not in (float, int) or not math.isfinite(count) or count < 0:
            raise ValueError("missing or invalid preemption counter")
    # Recompute from request rows and elapsed time; saved valid/aggregates are
    # evidence to check, not a substitute for complete input measurements.
    fresh = summarize(rows, sample["elapsed_s"], tokens, before, after)
    if not fresh["valid"] or fresh["preemptions"] != 0:
        raise ValueError("invalid measured preemptions")
    for key in ["completion_tok_s", "turn_p50_s", "turn_p90_s", "ttft_p50_s", "ttft_p90_s"]:
        if not positive(sample.get(key)) or not math.isclose(sample[key], fresh[key], rel_tol=1e-9):
            raise ValueError("saved aggregate differs from request measurements")


def compare(paths):
    if len(paths) != 4:
        raise ValueError("exactly four files in A/B/B/A order required")
    if len({Path(p).resolve() for p in paths}) != 4:
        raise ValueError("duplicate arm file")
    arms = [json.loads(Path(p).read_text()) for p in paths]
    reference = arms[0]
    if type(reference["settings"]["rounds"]) is not int or reference["settings"]["rounds"] < 3:
        raise ValueError("acceptance requires at least three rounds per stage")
    if reference["settings"].get("tokens") != 128:
        raise ValueError("acceptance requires 128 output tokens")
    ids, starts = [], []
    for arm in arms:
        manifest = arm["manifest"]
        for key in ["arm", "started_msk", "wheel_sha256", "model_revision", "gpu", "server_argv"]:
            if not present(manifest.get(key)):
                raise ValueError(f"missing identity: {key}")
        ids.append(manifest["arm"])
        started = datetime.datetime.fromisoformat(manifest["started_msk"])
        if started.tzinfo is None:
            raise ValueError("run start requires timezone")
        starts.append(started.timestamp())
        if type(manifest.get("seats")) is not int or manifest["seats"] not in [16, 24, 32]:
            raise ValueError("invalid seat limit")
        if type(manifest.get("qpn2")) is not bool:
            raise ValueError("QPN2 state missing")
        for key in ["wheel_sha256", "model_revision", "gpu"]:
            if arm["manifest"][key] != reference["manifest"][key]:
                raise ValueError(f"incomparable {key}")
        if arm["settings"] != reference["settings"]:
            raise ValueError("incomparable workload settings")
    if len(set(ids)) != 4 or len(set(starts)) != 4 or starts != sorted(starts):
        raise ValueError("distinct runs in chronological A/B/B/A order required")
    for first, second in [(0, 3), (1, 2)]:
        for key in ["seats", "qpn2", "server_argv"]:
            if arms[first]["manifest"][key] != arms[second]["manifest"][key]:
                raise ValueError(f"changed {key} inside repeated arm")
    expected = {(c, n) for c in reference["settings"]["contexts"]
                for n in reference["settings"]["clients"]}
    if expected != {(c, n) for c in [8000, 33000] for n in [16, 24, 32]}:
        raise ValueError("acceptance requires the full 16/24/32 by 8k/33k matrix")
    tables = []
    for arm in arms:
        table = {(s["context"], s["clients"]): s for s in arm["stages"]}
        if set(table) != expected or len(table) != len(arm["stages"]):
            raise ValueError("missing or duplicate stage")
        for stage in table.values():
            if (len(stage["rounds"]) != arm["settings"]["rounds"]
                    or not stage["warmup"]["valid"]):
                raise ValueError("incomplete stage")
            if not present(stage.get("prompt_sha256")):
                raise ValueError("missing prompt identity")
            checked_round(stage["warmup"], stage["context"], stage["clients"], 1)
            for sample in stage["rounds"]:
                checked_round(sample, stage["context"], stage["clients"], 128)
        tables.append(table)
    rows, passed = [], True
    for key in sorted(expected):
        if len({table[key]["prompt_sha256"] for table in tables}) != 1:
            raise ValueError("prompt mismatch")
        groups = []
        for indexes in [(0, 3), (1, 2)]:
            rounds = [r for index in indexes for r in tables[index][key]["rounds"]]
            durations = [r["turn_s"] for sample in rounds for r in sample["rows"]]
            groups.append({
                "tok_s": statistics.median(r["completion_tok_s"] for r in rounds),
                "p50": statistics.median(durations), "p90": percentile(durations, .9),
                "preemptions": sum(r["preemptions"] for r in rounds),
                "round_tok_s": [r["completion_tok_s"] for r in rounds],
            })
        base, candidate = groups
        ratios = {metric: candidate[metric] / base[metric] for metric in ["tok_s", "p50", "p90"]}
        ok = (ratios["p50"] <= 1.05 and ratios["p90"] <= 1.05
              and base["preemptions"] == 0 and candidate["preemptions"] == 0
              and (key[1] == 16 or ratios["tok_s"] >= 1.05))
        passed = passed and ok
        rows.append({"context": key[0], "clients": key[1], "base": base,
                     "candidate": candidate, "ratios": ratios, "performance_gate": ok})
    return {"performance_gate": passed, "quality_gate": "not evaluated: mixbench quick required",
            "rollout_approved": False, "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arms", nargs=4)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.write_text(json.dumps(compare(args.arms), indent=2) + "\n")
