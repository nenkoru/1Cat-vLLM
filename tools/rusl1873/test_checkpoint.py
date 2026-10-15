# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from concurrency import GAUGES, checkpoint, main


class CheckpointTest(unittest.TestCase):
    def test_failed_write_keeps_previous_valid_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"arm.json"
            previous = {"stages": [{"rounds": [{"valid": True}]}]}
            checkpoint(path, previous)

            def interrupted_dump(value, handle, **kwargs):
                handle.write('{"stages": [')
                raise OSError("simulated interrupted write")

            with patch("concurrency.json.dump", side_effect=interrupted_dump):
                with self.assertRaises(OSError):
                    checkpoint(path, {"stages": []})
            self.assertEqual(json.loads(path.read_text()), previous)
            self.assertEqual(list(Path(directory).glob('*.tmp')), [])
            checkpoint(path, {"stages": [], "new": True})
            self.assertEqual(json.loads(path.read_text()), {"stages": [], "new": True})

    def test_warmup_saved_when_first_measurement_is_interrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/"arm.json"
            manifest = Path(directory)/"manifest.json"
            manifest.write_text(json.dumps({k: "test" for k in
                ("arm", "wheel_sha256", "model_revision", "server_argv", "seats", "qpn2", "gpu")}))
            argv = ['concurrency.py', '--url', 'http://127.0.0.1:8013', '--model', 'test',
                    '--manifest', str(manifest), '--out', str(out), '--contexts', '3',
                    '--clients', '1', '--rounds', '1']
            warmup = {"valid": True, "rows": [{"completion_tokens": 1}]}
            with patch('sys.argv', argv), patch('concurrency.Client') as factory:
                client = factory.return_value
                client.metrics.return_value = dict.fromkeys(GAUGES, 0)
                client.json.return_value = {"tokens": [1, 2, 3]}
                client.round.side_effect = [warmup, RuntimeError('slot yielded')]
                with self.assertRaisesRegex(RuntimeError, 'slot yielded'):
                    main()
            saved = json.loads(out.read_text())
            self.assertEqual(saved['stages'][0]['warmup'], warmup)
            self.assertEqual(saved['stages'][0]['rounds'], [])


if __name__ == '__main__':
    unittest.main()
