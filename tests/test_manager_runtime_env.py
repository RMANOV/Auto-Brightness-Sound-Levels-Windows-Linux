"""Cron-like environment with a fake logind provider; no host session calls."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SessionRuntimeTests(unittest.TestCase):
    def check_runtime(self, provider, existing=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime"
            runtime.mkdir(mode=0o700)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            loginctl = bin_dir / "loginctl"
            loginctl.write_text('#!/bin/sh\nprintf "%s\\n" "$TEST_RUNTIME"\n')
            loginctl.chmod(0o700)
            source = (ROOT / "adaptive_controller_manager.sh").read_text()
            prefix, main = source.split("# Main logic", 1)
            override = '\nshould_be_active() { printf "%s" "${XDG_RUNTIME_DIR:-}" > "$TEST_RESULT"; return 1; }\n'
            env = {k: v for k, v in os.environ.items() if k != "XDG_RUNTIME_DIR"}
            env.update(HOME=tmp, XDG_STATE_HOME=str(root / "state"),
                       PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
                       TEST_RESULT=str(root / "result"),
                       TEST_RUNTIME=str(runtime) if provider == "owned" else provider)
            if existing is not None:
                env["XDG_RUNTIME_DIR"] = existing
            result = subprocess.run(["bash", "-c", prefix + override + "# Main logic" + main],
                                    cwd=ROOT, env=env, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return (root / "result").read_text(), str(runtime)

    def test_missing_cron_runtime_is_restored_from_owned_logind_path(self):
        actual, expected = self.check_runtime("owned")
        self.assertEqual(actual, expected)

    def test_explicit_session_runtime_is_preserved(self):
        actual, _ = self.check_runtime("owned", "/explicit/session")
        self.assertEqual(actual, "/explicit/session")

    def test_invalid_or_unowned_logind_path_is_not_exported(self):
        for value in ("", "relative/path", "/missing/session/runtime", "/"):
            with self.subTest(value=value):
                actual, _ = self.check_runtime(value)
                self.assertEqual(actual, "")


if __name__ == "__main__":
    unittest.main()
