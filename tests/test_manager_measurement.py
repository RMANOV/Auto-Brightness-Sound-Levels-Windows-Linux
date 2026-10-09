"""Exercise the manager with fake sensor/process boundaries, never host hardware."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ManagerMeasurementTests(unittest.TestCase):
    def flash(self, output, exit_code):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = (ROOT / "adaptive_controller_manager.sh").read_text().split("# Main logic", 1)[0]
            # The real producer boundary is replaced; validation and routing stay real.
            script = prefix + r'''
LOG_FILE="$TEST_DIR/log"
PROBE_FILE="$TEST_DIR/probe"
AMBIENT_STATE="$TEST_DIR/state"
python3() { printf '%s\n' "$TEST_OUTPUT"; return "$TEST_RC"; }
timeout() { shift 2; "$@"; }
log_message() { printf '%s\n' "$1"; }
flash_detection_check
'''
            import os
            env = {**os.environ, "TEST_DIR": tmp, "TEST_OUTPUT": output,
                   "TEST_RC": str(exit_code), "HOME": tmp}
            return subprocess.run(["bash", "-c", script], env=env,
                                  capture_output=True, text=True, timeout=5)

    def test_sensor_failure_is_error_not_unchanged(self):
        r = self.flash("flash_error:no_measurement_possible", 2)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("ERROR", r.stdout)
        self.assertNotIn("No significant change", r.stdout)

    def test_empty_success_is_invalid_measurement(self):
        r = self.flash("", 0)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("ERROR", r.stdout)

    def test_timeout_is_not_valid_unchanged(self):
        r = self.flash("", 124)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("ERROR", r.stdout)


if __name__ == "__main__":
    unittest.main()
