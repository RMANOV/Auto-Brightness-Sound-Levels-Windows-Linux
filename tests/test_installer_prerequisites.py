"""Actual installer route with a private checkout and recording crontab leaf."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("bash"), "Bash installer required")
class InstallerPrerequisiteTests(unittest.TestCase):
    def fixture(self, backend):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        checkout = root / "checkout with spaces"
        checkout.mkdir()
        shutil.copy2(ROOT / "install_crontab.sh", checkout)
        binary = checkout / "adaptive-rust/target/release/adaptive-controller"
        binary.parent.mkdir(parents=True)
        if backend == "directory":
            binary.mkdir()
        elif backend != "missing":
            binary.write_text('#!/bin/bash\nprintf invoked > "$TEST_BACKEND_CALLED"\nexit 91\n')
            binary.chmod(0o700 if backend == "executable" else 0o600)
        commands = root / "bin"
        commands.mkdir()
        crontab = commands / "crontab"
        crontab.write_text('''#!/bin/bash
printf '%s\\n' "$#" "$@" >> "$TEST_CRONTAB_CALLS"
[[ "$#" == "1" && -f "$1" ]] || exit 92
cp -- "$1" "$TEST_CRONTAB_STATE"
''')
        crontab.chmod(0o700)
        current = root / "crontab"
        current.write_text("# EXISTING FIXTURE CRONTAB\n")
        calls = root / "crontab-calls"
        backend_called = root / "backend-called"
        scratch = root / "tmp"
        scratch.mkdir()
        env = {**os.environ, "PATH": str(commands) + os.pathsep + os.environ["PATH"],
               "TMPDIR": str(scratch), "TEST_CRONTAB_CALLS": str(calls),
               "TEST_CRONTAB_STATE": str(current),
               "TEST_BACKEND_CALLED": str(backend_called)}
        result = subprocess.run(["bash", str(checkout / "install_crontab.sh")],
                                cwd=root, env=env, text=True, capture_output=True,
                                timeout=4)
        return root, checkout, current, calls, backend_called, scratch, result

    def assert_prerequisite_rejected(self, backend):
        _, checkout, current, calls, invoked, scratch, result = self.fixture(backend)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(calls.exists(), "Installer reached crontab despite missing prerequisite")
        self.assertEqual(current.read_text(), "# EXISTING FIXTURE CRONTAB\n")
        self.assertFalse(invoked.exists(), "Prerequisite validation executed the backend")
        self.assertEqual(list(scratch.iterdir()), [])
        self.assertIn(str(checkout / "adaptive-rust/target/release/adaptive-controller"), result.stderr)
        self.assertIn("cargo build --release -p adaptive-controller --locked", result.stderr)
        self.assertNotIn("Crontab installed successfully", result.stdout)

    def test_missing_backend_aborts_before_crontab_write(self):
        self.assert_prerequisite_rejected("missing")

    def test_nonexecutable_backend_aborts_before_crontab_write(self):
        self.assert_prerequisite_rejected("nonexecutable")

    def test_directory_is_not_an_executable_backend(self):
        self.assert_prerequisite_rejected("directory")

    def test_available_backend_preserves_existing_install_route(self):
        _, checkout, current, calls, invoked, scratch, result = self.fixture("executable")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        arguments = calls.read_text().splitlines()
        self.assertEqual(arguments[0], "1")
        self.assertEqual(len(arguments), 2)
        self.assertFalse(Path(arguments[1]).exists(), "Temporary crontab was not removed")
        self.assertFalse(invoked.exists(), "Installation must not run the hardware backend")
        self.assertEqual(list(scratch.iterdir()), [])
        installed = current.read_text()
        manager = checkout / "adaptive_controller_manager.sh"
        self.assertIn(f'*/10 * * * * DISPLAY=:0 "{manager}" check', installed)
        self.assertIn(f'0 3 * * * DISPLAY=:0 "{manager}" restart', installed)
        self.assertNotIn("EXISTING FIXTURE CRONTAB", installed)
        self.assertIn("Crontab installed successfully", result.stdout)


if __name__ == "__main__":
    unittest.main()
