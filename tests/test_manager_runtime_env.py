"""Cron-like environment with a fake logind provider; no host session calls."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SessionRuntimeTests(unittest.TestCase):
    def check_runtime(self, provider, existing=None, private_directory=None, mode=0o700):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime"
            runtime.mkdir(mode=mode)
            runtime.chmod(mode)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            loginctl = bin_dir / "loginctl"
            loginctl.write_text('#!/bin/sh\nprintf "%s\\n" "$*" > "$TEST_LOGIND_CALL"\n'
                                'printf "%s\\n" "$TEST_RUNTIME"\n')
            loginctl.chmod(0o700)
            source = (ROOT / "adaptive_controller_manager.sh").read_text()
            prefix, main = source.split("# Main logic", 1)
            override = '\nshould_be_active() { printf "%s" "${XDG_RUNTIME_DIR:-}" > "$TEST_RESULT"; return 1; }\n'
            if private_directory is not None:
                # Only the filesystem leaf is replaced. Canonical-path routing
                # and the full main block remain actual source. This also models
                # root ownership of '/' without becoming root or touching /run.
                override += '''
runtime_directory_is_private() {
    printf '%s\\n' "$1" >> "$TEST_DIRECTORY_CALL"
    [[ "$TEST_DIRECTORY_PRIVATE" == true ]]
}
'''
            env = {k: v for k, v in os.environ.items() if k != "XDG_RUNTIME_DIR"}
            env.update(HOME=tmp, XDG_STATE_HOME=str(root / "state"),
                       PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
                       TEST_RESULT=str(root / "result"),
                       TEST_LOGIND_CALL=str(root / "logind-call"),
                       TEST_DIRECTORY_CALL=str(root / "directory-call"),
                       TEST_DIRECTORY_PRIVATE=str(private_directory).lower(),
                       TEST_RUNTIME=(str(runtime) if provider == "owned" else
                                     f"/run/user/{os.getuid()}" if provider == "canonical" else provider))
            if existing is not None:
                env["XDG_RUNTIME_DIR"] = existing
            result = subprocess.run(["bash", "-c", prefix + override + "# Main logic" + main],
                                    cwd=ROOT, env=env, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return {"actual": (root / "result").read_text(),
                    "provided": env["TEST_RUNTIME"],
                    "logind_calls": (root / "logind-call").read_text()
                    if (root / "logind-call").exists() else "",
                    "directory_calls": (root / "directory-call").read_text()
                    if (root / "directory-call").exists() else ""}

    def test_missing_cron_runtime_is_restored_from_owned_logind_path(self):
        result = self.check_runtime("canonical", private_directory=True)
        self.assertEqual(result["actual"], result["provided"])
        self.assertEqual(result["directory_calls"], result["provided"] + "\n")
        self.assertEqual(result["logind_calls"], f"show-user {os.getuid()} -p RuntimePath --value\n")

    def test_explicit_session_runtime_is_preserved(self):
        result = self.check_runtime("owned", "/explicit/session")
        self.assertEqual(result["actual"], "/explicit/session")
        self.assertEqual(result["logind_calls"], "")

    def test_invalid_or_unowned_logind_path_is_not_exported(self):
        for value in ("", "relative/path", "/missing/session/runtime", "/"):
            with self.subTest(value=value):
                result = self.check_runtime(value, private_directory=True)
                self.assertEqual(result["actual"], "")
                self.assertEqual(result["directory_calls"], "")

    def test_arbitrary_owned_private_directory_is_not_a_session_runtime(self):
        for mode in (0o700, 0o755):
            with self.subTest(mode=oct(mode)):
                result = self.check_runtime("owned", mode=mode)
                self.assertEqual(result["actual"], "")

    def test_canonical_path_with_failed_filesystem_validation_is_rejected(self):
        result = self.check_runtime("canonical", private_directory=False)
        self.assertEqual(result["actual"], "")
        self.assertEqual(result["directory_calls"], result["provided"] + "\n")

    def test_actual_directory_leaf_requires_owned_directory_with_0700_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "runtime"
            directory.mkdir()
            regular_file = root / "file"
            regular_file.touch(mode=0o700)
            source = (ROOT / "adaptive_controller_manager.sh").read_text()
            prefix = source.split("# Main logic", 1)[0]
            for path, mode, expected in [(directory, 0o700, 0), (directory, 0o755, 1),
                                         (regular_file, 0o700, 1)]:
                with self.subTest(path=path.name, mode=oct(mode)):
                    path.chmod(mode)
                    result = subprocess.run(
                        ["bash", "-c", prefix + '\nruntime_directory_is_private "$TEST_DIRECTORY"'],
                        cwd=ROOT, env={**os.environ, "HOME": tmp, "TEST_DIRECTORY": str(path)},
                        capture_output=True, text=True, timeout=5)
                    self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                    self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
