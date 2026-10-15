# SPDX-License-Identifier: Apache-2.0
"""Measure warm full turns at fixed offered concurrency on an isolated server.

No server mutation, model download, or production control. Run on the allocated
benchmark host. SSE chunks are NOT tokens or engine steps. Throughput uses the
server's completion_tokens usage. Missing counters/usage invalidate a round.
"""
import argparse
import concurrent.futures
import hashlib
import http.client
import json
import math
from pathlib import Path
import random
import re
import statistics
import threading
import time
from urllib.parse import urlsplit


COUNTERS = ("vllm:num_preemptions_total",)
GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting")
WORDS = "ledger stock parcel route cache worker batch queue token order".split()


def metric_values(text, names):
    values = {name: None for name in names}
    for line in text.splitlines():
        match = re.match(r'^([^\s{]+)(?:\{[^}]*\})?\s+([^\s]+)', line)
        if match and match[1] in values:
            number = float(match[2])
            if not math.isfinite(number):
                raise ValueError("nonfinite server metric")
            values[match[1]] = (values[match[1]] or 0) + number
    return values


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def summarize(rows, elapsed, expected_tokens, before, after):
    errors = [r.get("error") for r in rows if r.get("error")]
    if not rows or elapsed <= 0:
        errors.append("empty round or nonpositive duration")
    if any(r.get("completion_tokens") != expected_tokens for r in rows):
        errors.append("missing/short completion usage")
    preemptions = None
    key = COUNTERS[0]
    if before.get(key) is None or after.get(key) is None:
        errors.append("preemption counter missing")
    elif after[key] < before[key]:
        errors.append("preemption counter reset; possible server restart")
    else:
        preemptions = after[key] - before[key]
    if errors:
        return {"valid": False, "errors": errors, "preemptions": preemptions}
    if any("first_s" not in r or "turn_s" not in r for r in rows):
        return {"valid": False, "errors": ["missing stream timings"]}
    return {
        "valid": True, "errors": [], "preemptions": preemptions,
        "completion_tok_s": sum(r["completion_tokens"] for r in rows) / elapsed,
        "turn_p50_s": statistics.median(r["turn_s"] for r in rows),
        "turn_p90_s": percentile([r["turn_s"] for r in rows], .9),
        "ttft_p50_s": statistics.median(r["first_s"] for r in rows),
        "ttft_p90_s": percentile([r["first_s"] for r in rows], .9),
    }


class Client:
    def __init__(self, url, model, timeout):
        self.url = urlsplit(url)
        if self.url.scheme != "http" or self.url.username or self.url.password:
            raise ValueError("use an unauthenticated HTTP benchmark endpoint")
        self.model, self.timeout = model, timeout

    def request(self, path, body=None):
        conn = http.client.HTTPConnection(self.url.hostname, self.url.port,
                                          timeout=self.timeout)
        try:
            conn.request("GET" if body is None else "POST",
                         self.url.path.rstrip("/") + path,
                         None if body is None else json.dumps(body),
                         {"Content-Type": "application/json"})
            response = conn.getresponse()
            if response.status != 200:
                raise RuntimeError(f"{path}: HTTP {response.status}")
            return conn, response
        except Exception:
            conn.close()
            raise

    def json(self, path, body=None):
        conn, response = self.request(path, body)
        try:
            return json.load(response)
        finally:
            conn.close()

    def metrics(self):
        conn, response = self.request("/metrics")
        try:
            return metric_values(response.read().decode(), COUNTERS + GAUGES)
        finally:
            conn.close()

    def ask(self, prompt, count, barrier):
        barrier.wait(timeout=self.timeout)
        start, conn = time.monotonic(), None
        row = {}
        digest = hashlib.sha256()
        try:
            conn, response = self.request("/v1/completions", {
                "model": self.model, "prompt": prompt, "max_tokens": count,
                "temperature": 0, "seed": 1873, "ignore_eos": True,
                "stream": True, "stream_options": {"include_usage": True},
            })
            done = False
            for raw in response:
                if time.monotonic() - start > self.timeout:
                    raise TimeoutError("request deadline")
                if not raw.startswith(b"data:"):
                    continue
                data = raw[5:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise RuntimeError("stream error event")
                for choice in event.get("choices", []):
                    text = choice.get("text") or ""
                    if text:
                        row.setdefault("first_s", time.monotonic() - start)
                        digest.update(text.encode())
                if event.get("usage"):
                    usage = event["usage"]
                    row["completion_tokens"] = usage.get("completion_tokens")
                    row["prompt_tokens"] = usage.get("prompt_tokens")
                    row["cached_tokens"] = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            if not done:
                raise RuntimeError("stream ended without DONE")
            row["turn_s"] = time.monotonic() - start
            row["output_sha256"] = digest.hexdigest()
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if conn is not None:
                conn.close()
        return row

    def round(self, prompts, count):
        barrier = threading.Barrier(len(prompts) + 1)
        before = self.metrics()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            jobs = [pool.submit(self.ask, p, count, barrier) for p in prompts]
            start = time.monotonic()
            barrier.wait(timeout=self.timeout)
            rows = [j.result() for j in jobs]
            elapsed = time.monotonic() - start
        after = self.metrics()
        result = summarize(rows, elapsed, count, before, after)
        if any(r.get("prompt_tokens") != len(p) for r, p in zip(rows, prompts)):
            result["valid"] = False
            result["errors"].append("prompt usage differs from submitted token IDs")
        result.update(rows=rows, elapsed_s=elapsed, metrics_before=before,
                      metrics_after=after)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="verified server argv, wheel/model hashes, seats, QPN2, GPU")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--contexts", nargs="+", type=int, default=[8000, 33000])
    parser.add_argument("--clients", nargs="+", type=int, default=[16, 24, 32])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if min(args.contexts + args.clients + [args.rounds, args.tokens, args.timeout]) <= 0:
        parser.error("all sizes and timeouts must be positive")
    if max(args.clients) > 32:
        parser.error("this experiment is limited to 32 clients")
    if args.out.exists():
        parser.error("output exists; choose a new path")
    manifest = json.loads(args.manifest.read_text())
    required = {"arm", "wheel_sha256", "model_revision", "server_argv", "seats", "qpn2", "gpu"}
    if required - manifest.keys():
        parser.error(f"manifest missing {sorted(required - manifest.keys())}")
    client = Client(args.url, args.model, args.timeout)
    initial = client.metrics()
    if any(initial.get(k) != 0 for k in GAUGES):
        parser.error("server not idle, or idle metrics missing")
    result = {"manifest": manifest, "timing": "full turn; not engine step",
              "settings": {"contexts": args.contexts, "clients": args.clients,
                           "rounds": args.rounds, "tokens": args.tokens}, "stages": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    valid = True
    for context in args.contexts:
        # Same prompt IDs across arms and offered concurrency. No private data.
        prompts = []
        for index in range(max(args.clients)):
            rng = random.Random(f"rusl1873-{context}-{index}")
            text = " ".join(rng.choice(WORDS) for _ in range(context + 128))
            ids = client.json("/tokenize", {"model": args.model, "prompt": text})["tokens"]
            if len(ids) < context:
                raise ValueError("tokenized corpus shorter than requested context")
            prompts.append(ids[:context])
        for count in args.clients:
            selected = prompts[:count]
            stage = {"context": context, "clients": count,
                     "prompt_sha256": hashlib.sha256(json.dumps(selected).encode()).hexdigest(),
                     "warmup": client.round(selected, 1), "rounds": []}
            result["stages"].append(stage)
            valid = valid and stage["warmup"]["valid"]
            if stage["warmup"]["valid"]:
                for _ in range(args.rounds):
                    measured = client.round(selected, args.tokens)
                    stage["rounds"].append(measured)
                    valid = valid and measured["valid"]
                    args.out.write_text(json.dumps(result, indent=2) + "\n")
                    if not measured["valid"]:
                        break
            args.out.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({"context": context, "clients": count, "valid_so_far": valid}), flush=True)
            if not valid:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
