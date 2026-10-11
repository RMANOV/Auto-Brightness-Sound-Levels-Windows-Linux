"""Actual scripts/update-local.sh with recording git, cargo, crontab and manager leaves."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]

LEAVES = {
    "bin/git": '''#!/bin/bash
printf 'git %s\\n' "$*" >> "$TEST_CALLS"
case "$*" in
    *rev-parse*) printf '%s\\n' "$TEST_BRANCH" ;;
    *merge*) exit "$TEST_MERGE_RC" ;;
esac
''',
    "bin/cargo": '''#!/bin/bash
printf 'cargo %s in %s\\n' "$*" "$(basename "$PWD")" >> "$TEST_CALLS"
''',
    "bin/crontab": '''#!/bin/bash
printf 'crontab %s\\n' "$*" >> "$TEST_CALLS"
[[ "$1" == "-l" && -n "$TEST_CRON" ]] && printf '%s\\n' "$TEST_CRON"
exit 0
''',
    "checkout/adaptive_controller_manager.sh": '''#!/bin/bash
printf 'manager %s\\n' "$*" >> "$TEST_CALLS"
exit "$TEST_MANAGER_RC"
''',
}


@unittest.skipUnless(shutil.which("bash"), "Bash required")
class UpdateLocalTests(unittest.TestCase):
    def run_update(self, branch="main", merge_rc=0, manager_rc=0, cron="*/10"):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, body in LEAVES.items():
                leaf = root / name
                leaf.parent.mkdir(parents=True, exist_ok=True)
                leaf.write_text(body)
                leaf.chmod(0o700)
            checkout = root / "checkout"
            (checkout / "scripts").mkdir()
            (checkout / "adaptive-rust").mkdir()
            shutil.copy2(ROOT / "scripts/update-local.sh", checkout / "scripts")
            manager = checkout / "adaptive_controller_manager.sh"
            calls = root / "calls"
            env = {**os.environ, "PATH": str(root / "bin") + os.pathsep + os.environ["PATH"],
                   "HOME": tmp, "TEST_CALLS": str(calls), "TEST_BRANCH": branch,
                   "TEST_MERGE_RC": str(merge_rc), "TEST_MANAGER_RC": str(manager_rc),
                   "TEST_CRON": f'{cron} * * * * DISPLAY=:0 "{manager}" check' if cron else ""}
            result = subprocess.run(["bash", str(checkout / "scripts/update-local.sh")],
                                    env=env, capture_output=True, text=True, timeout=10)
            lines = calls.read_text().splitlines() if calls.exists() else []
            return result, lines, str(checkout)

    def test_pulls_builds_restarts_and_checks_cron_without_rewriting_it(self):
        result, calls, checkout = self.run_update()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(calls, [
            f"git -C {checkout} fetch origin main",
            f"git -C {checkout} rev-parse --abbrev-ref HEAD",
            f"git -C {checkout} merge --ff-only origin/main",
            "cargo build --release -p adaptive-controller --locked in adaptive-rust",
            "manager restart",
            "crontab -l",
        ])
        self.assertIn("OK: cron runs the manager every 10 minutes", result.stdout)

    def test_other_branch_switches_to_main_first(self):
        result, calls, checkout = self.run_update(branch="feature")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"git -C {checkout} checkout main", calls)
        self.assertLess(calls.index(f"git -C {checkout} checkout main"),
                        calls.index(f"git -C {checkout} merge --ff-only origin/main"))

    def test_failed_fast_forward_stops_before_build_and_restart(self):
        result, calls, _ = self.run_update(merge_rc=1)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse([c for c in calls if c.startswith(("cargo", "manager"))], calls)

    def test_restart_outcomes(self):
        for manager_rc, expected_rc, message in [
                (1, 0, "no adjustment needed right now"),
                (2, 1, "Restart failed (exit 2)")]:
            with self.subTest(manager_rc=manager_rc):
                result, _, _ = self.run_update(manager_rc=manager_rc)
                self.assertEqual(result.returncode, expected_rc, result.stdout + result.stderr)
                self.assertIn(message, result.stdout + result.stderr)

    def test_missing_old_or_disabled_cron_entry_is_reported_not_installed(self):
        # install_crontab.sh wrote */30 before 2026-10-04; a disabled line does not run.
        for cron in ("", "*/30", "# */10"):
            with self.subTest(cron=cron):
                result, calls, _ = self.run_update(cron=cron)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("No 10-minute cron entry", result.stderr)
                self.assertNotIn("OK: cron", result.stdout)
                self.assertEqual([c for c in calls if c.startswith("crontab")], ["crontab -l"])


if __name__ == "__main__":
    unittest.main()
