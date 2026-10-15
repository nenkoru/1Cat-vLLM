# SPDX-License-Identifier: Apache-2.0
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import unittest

from concurrency import Client, COUNTERS, metric_values, summarize
from profile_concurrency import ObservedClient


class Handler(BaseHTTPRequestHandler):
    # One SSE content event contains many tokens. Tests must use usage, not chunks.
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'vllm:num_preemptions_total{model_name="test"} 0\n')

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.send_response(200)
        self.end_headers()
        events = [
            {"choices": [{"text": "many tokens in one chunk"}]},
            {"choices": [], "usage": {"completion_tokens": body["max_tokens"],
                                      "prompt_tokens": len(body["prompt"])}}
        ]
        for event in events:
            self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")


class ConcurrencyTest(unittest.TestCase):
    def test_usage_counts_not_stream_chunks(self):
        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                client = Client(f"http://127.0.0.1:{server.server_port}", "test", 5)
                result = client.round([[1, 2, 3], [4, 5, 6]], 128)
                self.assertTrue(result["valid"], result)
                self.assertAlmostEqual(result["completion_tok_s"] * result["elapsed_s"], 256)
                self.assertEqual(len(result["rows"]), 2)
                self.assertEqual(result["preemptions"], 0)
            finally:
                server.shutdown()
                thread.join()

    def test_metrics_missing_is_not_zero(self):
        values = metric_values("# HELP x y\nvllm:num_preemptions_total{rank=\"0\"} 3\n"
                               "vllm:num_preemptions_total{rank=\"1\"} 2\n", COUNTERS + ("absent",))
        self.assertEqual(values[COUNTERS[0]], 5)
        self.assertIsNone(values["absent"])

    def test_profile_observer_tracks_each_started_and_finished_stream(self):
        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                client = ObservedClient(f"http://127.0.0.1:{server.server_port}", "test", 5)
                result = client.round([[1, 2], [3, 4]], 128)
                self.assertTrue(result["valid"], result)
                self.assertEqual(len(client.first_threads), 2)
                self.assertEqual(len(client.finished_threads), 2)
            finally:
                server.shutdown()
                thread.join()

    def test_missing_or_short_usage_fails(self):
        for row in [{}, {"completion_tokens": 127}, {"error": "HTTP 500"}]:
            result = summarize([row], 1, 128, {COUNTERS[0]: 0}, {COUNTERS[0]: 0})
            self.assertFalse(result["valid"])
            self.assertNotIn("completion_tok_s", result)

    def test_counter_missing_or_reset_fails(self):
        row = {"completion_tokens": 128, "first_s": .1, "turn_s": 1}
        for before, after in [({}, {}), ({COUNTERS[0]: 5}, {COUNTERS[0]: 0})]:
            self.assertFalse(summarize([row], 1, 128, before, after)["valid"])

    def test_preemption_and_tail_latency_visible(self):
        rows = [{"completion_tokens": 10, "first_s": .1, "turn_s": x} for x in [1, 2, 9]]
        result = summarize(rows, 10, 10, {COUNTERS[0]: 1}, {COUNTERS[0]: 3})
        self.assertEqual(result["preemptions"], 2)
        self.assertEqual(result["turn_p50_s"], 2)
        self.assertEqual(result["turn_p90_s"], 9)
        self.assertEqual(result["completion_tok_s"], 3)


if __name__ == "__main__":
    unittest.main()
