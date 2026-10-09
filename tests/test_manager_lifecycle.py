"""The actual shell supervisor owns fake children; no controller hardware runs."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LifecycleTests(unittest.TestCase):
    def run_controller(self, body, timeout="3s", inspect_live=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = root / "backend"
            backend.write_text("#!/bin/bash\n" + body + "\n")
            backend.chmod(0o700)
            record = {"status": "CHANGED", "sample": {"schema": 1,
                      "source": "v4l2-gray-warmup3-median3-v1", "ambient": 40,
                      "captured_at": time.time()}}
            (root / "probe").write_text(json.dumps(record))
            prefix = (ROOT / "adaptive_controller_manager.sh").read_text().split("# Main logic", 1)[0]
            script = prefix + r'''
LOG_FILE="$TEST_DIR/log"
RUN_TMP_DIR="$TEST_DIR"
LOCK_FILE="$TEST_DIR/lock"
PID_FILE="$TEST_DIR/pid"
PROBE_FILE="$TEST_DIR/probe"
AMBIENT_STATE="$TEST_DIR/state"
RUST_BINARY="$TEST_DIR/backend"
USE_RUST=true
CONTROLLER_TIMEOUT="$TEST_TIMEOUT"
is_controller_running() { return 1; }
system_load_acceptable() { return 0; }
start_controller true
'''
            env = {**os.environ, "TEST_DIR": tmp, "TEST_TIMEOUT": timeout}
            proc = subprocess.Popen(["bash", "-c", script], cwd=ROOT, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                if inspect_live:
                    deadline = time.monotonic() + 1
                    while not (root / "pid").exists() and time.monotonic() < deadline:
                        time.sleep(.01)
                    self.assertTrue((root / "pid").exists())
                    time.sleep(.05)
                    self.assertIsNone(proc.poll(), "Supervisor exited while its child was alive")
                    self.assertTrue((root / "pid").exists(), "PID cleared before completion")
                stdout, stderr = proc.communicate(timeout=5)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
            log = (root / "log").read_text()
            self.assertFalse((root / "pid").exists())
            self.assertFalse((root / "lock").exists())
            return proc.returncode, log, (root / "state").exists(), stdout + stderr

    def test_parent_waits_and_accepts_reference_only_after_completion(self):
        rc, log, state, errors = self.run_controller('sleep .2; echo "Converged in .2s"', inspect_live=True)
        self.assertEqual(rc, 0, errors + log)
        self.assertTrue(state)
        self.assertNotIn("exit code: 127", log)

    def test_fast_success_is_not_mistaken_for_start_failure(self):
        rc, log, state, errors = self.run_controller('echo "Converged in .0s"')
        self.assertEqual(rc, 0, errors + log)
        self.assertTrue(state)

    def test_failure_no_convergence_and_timeout_preserve_reference(self):
        for body, timeout, expected in [('exit 7', '3s', 7), ('exit 0', '3s', 2),
                                        ('echo "Converged in .0s"; sleep 2', '.1s', 124)]:
            with self.subTest(body=body):
                rc, log, state, errors = self.run_controller(body, timeout)
                self.assertEqual(rc, expected, errors + log)
                self.assertFalse(state)
                self.assertIn("ERROR", log)


if __name__ == "__main__": unittest.main()
