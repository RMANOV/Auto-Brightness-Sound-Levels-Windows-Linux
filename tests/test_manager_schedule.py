"""Actual should_be_active policy with synthetic calculator, session and probe leaves.

Sunrise/sunset windows keep their checks, and every 10-minute cron check between
them also reaches the ambient probe. No camera, session or display is touched.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]

POLICY_HARNESS = r'''
SCRIPT_DIR="$TEST_DIR"
LOG_FILE="$TEST_DIR/log"
flash_detection_check() { printf probed > "$TEST_DIR/probed"; return "$TEST_PROBE_RC"; }
should_be_active
'''

LEAVES = {
    "sunrise_sunset_calculator.py": '''#!/bin/sh
[ "$1" = --check-active ] || exit 64
printf '%s\\n' "$TEST_CALC_OUTPUT"
exit "$TEST_CALC_RC"
''',
    "bin/loginctl": '''#!/bin/sh
case "$1" in
    list-sessions) printf '  7 %s %s seat0\\n' "$(id -u)" "$(whoami)" ;;
    show-session) printf 'Active=%s\\n' "$TEST_SESSION_ACTIVE" ;;
esac
''',
    "bin/xset": '''#!/bin/sh
if [ "$TEST_DISPLAY" = off ]; then echo '  Monitor is Off'; else echo '  Monitor is On'; fi
''',
}


class SchedulePolicyTests(unittest.TestCase):
    def policy(self, output, calc_rc=0, session="yes", display="on", probe_rc=0):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, body in LEAVES.items():
                leaf = root / name
                leaf.parent.mkdir(exist_ok=True)
                leaf.write_text(body)
                leaf.chmod(0o700)
            env = {**os.environ, "HOME": tmp, "XDG_STATE_HOME": str(root / "state"),
                   "PATH": str(root / "bin") + os.pathsep + os.environ["PATH"],
                   "TEST_DIR": tmp, "TEST_CALC_OUTPUT": output, "TEST_CALC_RC": str(calc_rc),
                   "TEST_SESSION_ACTIVE": session, "TEST_DISPLAY": display,
                   "TEST_PROBE_RC": str(probe_rc)}
            prefix = (ROOT / "adaptive_controller_manager.sh").read_text().split("# Main logic", 1)[0]
            result = subprocess.run(["bash", "-c", prefix + POLICY_HARNESS], cwd=ROOT, env=env,
                                    capture_output=True, text=True, timeout=5)
            log = (root / "log").read_text() if (root / "log").exists() else ""
            return result.returncode, log, (root / "probed").exists()

    def test_windows_and_ten_minute_periods_between_them_reach_the_probe(self):
        for output, logged in [("ACTIVE:sunrise", "Currently in sunrise activation window"),
                               ("ACTIVE:sunset", "Currently in sunset activation window"),
                               ("INACTIVE", "Between activation windows - 10-minute period check")]:
            for probe_rc in (0, 1, 2):  # CHANGED, UNCHANGED (>40% gate kept), probe ERROR
                with self.subTest(output=output, probe_rc=probe_rc):
                    rc, log, probed = self.policy(output, probe_rc=probe_rc)
                    self.assertEqual(rc, probe_rc, log)
                    self.assertTrue(probed, log)
                    self.assertIn(logged, log)
                    self.assertNotIn("skipping", log)

    def test_schedule_errors_still_defer_without_probing(self):
        for output, calc_rc in [("INACTIVE", 1), ("ACTIVE:sunrise", 2), ("", 0), ("garbage", 0)]:
            with self.subTest(output=output, calc_rc=calc_rc):
                rc, log, probed = self.policy(output, calc_rc=calc_rc)
                self.assertEqual(rc, 2, log)
                self.assertFalse(probed)
                self.assertIn("Schedule ERROR", log)

    def test_inactive_session_or_display_off_still_skip_between_windows(self):
        for session, display, logged in [("no", "on", "User session not active - skipping"),
                                         ("yes", "off", "Display is off - skipping")]:
            with self.subTest(session=session, display=display):
                rc, log, probed = self.policy("INACTIVE", session=session, display=display)
                self.assertEqual(rc, 1, log)
                self.assertFalse(probed)
                self.assertIn(logged, log)


if __name__ == "__main__":
    unittest.main()
