# SPDX-License-Identifier: Apache-2.0
"""Evaluate complete A/B/B/A arm files; never certify partial observations."""
import argparse
import json
from pathlib import Path
import statistics

from concurrency import percentile


def compare(paths):
    arms = [json.loads(Path(p).read_text()) for p in paths]
    if len(arms) != 4:
        raise ValueError("exactly four files in A/B/B/A order required")
    reference = arms[0]
    if reference["settings"]["rounds"] < 3:
        raise ValueError("acceptance requires at least three rounds per stage")
    for arm in arms:
        for key in ["wheel_sha256", "model_revision", "gpu"]:
            if arm["manifest"][key] != reference["manifest"][key]:
                raise ValueError(f"incomparable {key}")
        if arm["settings"] != reference["settings"]:
            raise ValueError("incomparable workload settings")
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
            if stage["warmup"].get("preemptions") != 0:
                raise ValueError("warmup preemptions missing or nonzero")
            if any(not r["valid"] for r in stage["rounds"]):
                raise ValueError("invalid measurement round")
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
