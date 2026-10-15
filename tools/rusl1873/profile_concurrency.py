# SPDX-License-Identifier: Apache-2.0
"""Capture a bounded profiler window while every offered stream is decoding.

Run separately from the unprofiled latency matrix. Server must have torch
profiler configured with max_iterations=40 and a dedicated output directory.
"""
import argparse
import concurrent.futures
import json
from pathlib import Path
import random
import shutil
import threading
import time

from concurrency import Client, WORDS


class ObservedClient(Client):
    def __init__(self, *args):
        super().__init__(*args)
        self.first_threads = set()
        self.finished_threads = set()
        self.lock = threading.Lock()

    def request(self, path, body=None):
        conn, response = super().request(path, body)
        if path != "/v1/completions":
            return conn, response
        observed = self

        class Stream:
            def __iter__(self):
                for raw in response:
                    if raw.startswith(b"data:"):
                        if raw[5:].strip() == b"[DONE]":
                            with observed.lock:
                                observed.finished_threads.add(threading.get_ident())
                        else:
                            event = json.loads(raw[5:])
                            if any(c.get("text") for c in event.get("choices", [])):
                                with observed.lock:
                                    observed.first_threads.add(threading.get_ident())
                    yield raw
        return conn, Stream()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile-dir", required=True, type=Path)
    parser.add_argument("--clients", required=True, type=int)
    parser.add_argument("--seats", required=True, type=int)
    parser.add_argument("--context", required=True, type=int)
    parser.add_argument("--name", required=True)
    parser.add_argument("--timeout", type=int, default=1800,
                        help="request deadline including queued cold prefill, seconds")
    args = parser.parse_args()
    if not 0 < args.clients <= args.seats <= 32:
        parser.error("profile requires all clients to fit into the seat limit")
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    dest = args.profile_dir / args.name
    dest.mkdir(exist_ok=False)
    before = set(args.profile_dir.glob("*.json*"))
    client = ObservedClient(args.url, args.model, args.timeout)
    prompts = []
    for index in range(args.clients):
        rng = random.Random(f"rusl1873-{args.context}-{index}")
        text = " ".join(rng.choice(WORDS) for _ in range(args.context + 128))
        ids = client.json("/tokenize", {"model": args.model, "prompt": text})["tokens"]
        if len(ids) < args.context:
            raise ValueError("short tokenized prompt")
        prompts.append(ids[:args.context])
    warmup = client.round(prompts, 1)
    (dest / "warmup.json").write_text(json.dumps(warmup, indent=2))
    if not warmup["valid"]:
        raise RuntimeError(f"warmup invalid: {warmup['errors']}")
    client.first_threads.clear()
    client.finished_threads.clear()
    started = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(client.round, prompts, 1024)
        start = time.monotonic()
        while len(client.first_threads) < args.clients:
            if future.done() or time.monotonic() - start > 120:
                raise RuntimeError("no common decode window; no profile claimed")
            time.sleep(.05)
        time.sleep(.5)
        state = {"clients": args.clients, "seats": args.seats, "context": args.context,
                 "request_timeout_s": args.timeout,
                 "streams_started": len(client.first_threads),
                 "metrics_at_start": client.metrics()}
        try:
            if future.done() or client.finished_threads:
                raise RuntimeError("streams finished before profiling")
            conn, response = client.request("/start_profile", {})
            response.read()
            conn.close()
            started = True
            time.sleep(10)
            state["streams_finished_before_stop"] = len(client.finished_threads)
        finally:
            if started:
                conn, response = client.request("/stop_profile", {})
                response.read()
                conn.close()
        state["round"] = future.result()
    for path in set(args.profile_dir.glob("*.json*")) - before:
        shutil.move(str(path), dest / path.name)
    state["traces"] = [p.name for p in dest.glob("*.pt.trace.json*")]
    (dest / "window.json").write_text(json.dumps(state, indent=2))
    if not state["traces"] or state["streams_finished_before_stop"] or not state["round"]["valid"]:
        raise RuntimeError("invalid profile window; inspect saved evidence")
    print(json.dumps({k: state[k] for k in ["context", "clients", "traces"]}), flush=True)


if __name__ == "__main__":
    main()
