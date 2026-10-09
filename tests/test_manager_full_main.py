"""Real manager dispatch, locks and traps with isolated synthetic leaf providers.

The complete main block and process-identity checks run unchanged. No camera,
session, load or brightness provider is called. Every spawned process belongs to
this fixture and is cleaned up even when an assertion fails.
"""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]


def process_info(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return {"state": fields[0], "ppid": int(fields[1]),
                "pgid": int(fields[2]), "start": int(fields[19])}
    except (FileNotFoundError, ProcessLookupError):
        return None


def wait_for(predicate, description, timeout=4):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(.02)
    raise AssertionError(f"Timed out waiting for {description}")


LEAF_OVERRIDES = r'''
SCRIPT_DIR="$TEST_SOURCE_DIR"
RUST_BINARY="$TEST_BACKEND"
USE_RUST=true
should_be_active() {
    PROBE_FILE=$(mktemp "$RUN_TMP_DIR/probe.XXXXXXXX") || return 2
    cp "$TEST_PROBE" "$PROBE_FILE" || return 2
    return 0
}
system_load_acceptable() { return 0; }
'''


BACKEND_BODY = r'''
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

mode = os.environ["TEST_BACKEND_MODE"]
if mode == "ignore":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

def announce(role):
    fields = Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(") ", 1)[1].split()
    record = {"role": role, "pid": os.getpid(), "pgid": os.getpgrp(),
              "start": int(fields[19])}
    with open(os.environ["TEST_EVENTS"], "a") as stream:
        stream.write(json.dumps(record) + "\n")

announce("backend")
if mode == "restart":
    try:
        fd = os.open(os.environ["TEST_FIRST_LAUNCH"], os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
    except FileExistsError:
        print("Converged in 0.0s", flush=True)
        sys.exit(0)
elif mode == "ignore":
    child_source = "\n".join([
        "import json, os, signal, time",
        "from pathlib import Path",
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
        "fields = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(') ', 1)[1].split()",
        "record = {'role': 'descendant', 'pid': os.getpid(), 'pgid': os.getpgrp(), 'start': int(fields[19])}",
        "with open(os.environ['TEST_EVENTS'], 'a') as stream: stream.write(json.dumps(record) + '\\n')",
        "while True: time.sleep(1)",
    ])
    subprocess.Popen([sys.executable, "-c", child_source])
while True:
    time.sleep(1)
'''


class ManagerFixture:
    def __init__(self, root, mode="hold", timeout="30s"):
        self.root = root
        runtime = root / "runtime"
        runtime.mkdir(mode=0o700)
        self.state = root / "state" / "adaptive-controller"
        self.state.mkdir(parents=True)
        self.pid_file = self.state / "controller.pid"
        self.lock_file = self.state / "controller.lock"
        self.ambient = root / "home/.config/adaptive-controller/ambient_state.json"
        self.events_file = root / "events.jsonl"
        self.backend = root / "backend"
        self.backend.write_text(f"#!{sys.executable}\n" + BACKEND_BODY)
        self.backend.chmod(0o700)
        probe = root / "probe.json"
        probe.write_text(json.dumps({"status": "CHANGED", "sample": {
            "schema": 1, "source": "v4l2-gray-warmup3-median3-v1",
            "ambient": 40, "captured_at": time.time()}}))
        source = (ROOT / "adaptive_controller_manager.sh").read_text()
        prefix, main = source.split("# Main logic", 1)
        self.script = root / "manager-under-test.sh"
        self.script.write_text(prefix + LEAF_OVERRIDES + "\n# Main logic" + main)
        self.env = {**os.environ, "HOME": str(root / "home"),
                    "XDG_STATE_HOME": str(root / "state"),
                    "XDG_RUNTIME_DIR": str(runtime),
                    "TEST_SOURCE_DIR": str(ROOT), "TEST_BACKEND": str(self.backend),
                    "TEST_PROBE": str(probe), "TEST_EVENTS": str(self.events_file),
                    "TEST_FIRST_LAUNCH": str(root / "first-launch"),
                    "TEST_BACKEND_MODE": mode, "ADAPTIVE_CONTROLLER_TIMEOUT": timeout}
        self.processes = []
        self.groups = {}

    def spawn(self, argv):
        proc = subprocess.Popen(argv, env=self.env, cwd=self.root,
                                start_new_session=True, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        self.processes.append(proc)
        info = process_info(proc.pid)
        if info:
            self.groups.setdefault(proc.pid, []).append((proc.pid, info["start"]))
        return proc

    def manager(self, command):
        return self.spawn(["bash", str(self.script), command])

    def events(self):
        if not self.events_file.exists():
            return []
        lines = self.events_file.read_text().splitlines()
        # An observer can catch the final write in progress; retry on that line.
        records = []
        for line in lines:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                break
        return records

    def running_supervisor(self):
        try:
            pid, start = map(int, self.pid_file.read_text().split())
            info = process_info(pid)
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")[:-1]
        except (FileNotFoundError, ProcessLookupError, ValueError):
            return None
        if (info and info["start"] == start and info["pgid"] == pid
                and argv[1:] == [b"--kill-after=2s",
                                 self.env["ADAPTIVE_CONTROLLER_TIMEOUT"].encode(),
                                 os.fsencode(self.backend)]):
            self.groups.setdefault(pid, []).append((pid, start))
            return pid
        return None

    def assert_events_dead(self, testcase):
        for event in self.events():
            wait_for(lambda: self.event_stopped(event), f"process {event} to stop", timeout=1)
            testcase.assertTrue(self.event_stopped(event), event)

    @staticmethod
    def event_stopped(event):
        info = process_info(event["pid"])
        return info is None or info["start"] != event["start"] or info["state"] == "Z"

    def close(self):
        # First let the actual EXIT trap clean up its child. Escalation here is
        # fixture-only and always targets a process/session created by this test.
        for proc in self.processes:
            if proc.poll() is None:
                proc.terminate()
        for proc in self.processes:
            try:
                proc.communicate(timeout=4)
            except subprocess.TimeoutExpired:
                proc.kill()
        for event in self.events():
            self.groups.setdefault(event["pgid"], []).append((event["pid"], event["start"]))
        for group, witnesses in self.groups.items():
            for witness, generation in witnesses:
                info = process_info(witness)
                if info and info["start"] == generation and info["pgid"] == group:
                    try:
                        os.killpg(group, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    break
        for proc in self.processes:
            proc.communicate(timeout=3)
        wait_for(lambda: all(self.event_stopped(event) for event in self.events()),
                 "fixture-owned descendants to stop", timeout=2)


@unittest.skipUnless(sys.platform == "linux" and shutil.which("timeout")
                     and shutil.which("setsid") and shutil.which("flock"),
                     "Linux process identity and GNU shell supervisors required")
class FullMainTests(unittest.TestCase):
    def fixture(self, mode="hold", timeout="30s"):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        fixture = ManagerFixture(Path(directory.name), mode, timeout)
        self.addCleanup(fixture.close)
        return fixture

    def assert_clean(self, fixture):
        self.assertFalse(fixture.pid_file.exists())
        self.assertFalse(fixture.lock_file.exists())
        self.assertEqual(list((fixture.state / "tmp").iterdir()), [])
        fixture.assert_events_dead(self)

    def test_active_restart_stops_first_and_starts_second_backend(self):
        fixture = self.fixture("restart")
        first = fixture.manager("start")
        first_supervisor = wait_for(fixture.running_supervisor, "first supervisor")
        wait_for(lambda: fixture.events(), "first backend")
        self.assertIsNone(first.poll())
        self.assertFalse(fixture.ambient.exists())
        status = fixture.manager("status")
        output, errors = status.communicate(timeout=2)
        self.assertEqual(status.returncode, 0, output + errors)
        self.assertIn(str(first_supervisor), output)
        restart = fixture.manager("restart")
        output, errors = restart.communicate(timeout=7)
        self.assertEqual(restart.returncode, 0, output + errors)
        first.communicate(timeout=2)
        self.assertNotEqual(first.returncode, 0)
        events = fixture.events()
        self.assertEqual(len(events), 2, events)
        self.assertNotEqual(events[0]["pid"], events[1]["pid"])
        self.assertEqual(json.loads(fixture.ambient.read_text())["ambient"], 40)
        self.assert_clean(fixture)

    def test_term_exit_trap_kills_term_ignoring_owned_group_within_bound(self):
        fixture = self.fixture("ignore")
        manager = fixture.manager("start")
        supervisor = wait_for(fixture.running_supervisor, "owned supervisor")
        wait_for(lambda: len(fixture.events()) == 2, "backend and descendant")
        self.assertTrue(all(event["pgid"] == supervisor for event in fixture.events()))
        started = time.monotonic()
        manager.send_signal(signal.SIGTERM)
        output, errors = manager.communicate(timeout=6)
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(manager.returncode, 143, output + errors)
        self.assertFalse(fixture.ambient.exists())
        self.assert_clean(fixture)

    def test_backend_deadline_forces_ignoring_group_and_rejects_acceptance(self):
        fixture = self.fixture("ignore", "0.5s")
        started = time.monotonic()
        manager = fixture.manager("check")
        wait_for(fixture.running_supervisor, "deadline supervisor")
        wait_for(lambda: len(fixture.events()) == 2, "ignoring descendant")
        output, errors = manager.communicate(timeout=6)
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(manager.returncode, 137, output + errors)
        self.assertFalse(fixture.ambient.exists())
        self.assertIn("no confirmed convergence", (fixture.state / "controller.log").read_text())
        self.assert_clean(fixture)

    def test_unrelated_shell_with_backend_argument_is_not_signalled(self):
        fixture = self.fixture()
        unrelated = fixture.spawn(["bash", "-c", 'read -r -t 30 ignored; : "$1"',
                                   "unrelated-shell", str(fixture.backend)])
        generation = process_info(unrelated.pid)["start"]
        fixture.pid_file.write_text(f"{unrelated.pid} {generation}\n")
        stop = fixture.manager("stop")
        output, errors = stop.communicate(timeout=3)
        self.assertEqual(stop.returncode, 0, output + errors)
        self.assertIsNone(unrelated.poll(), "Unrelated shell was signalled")
        self.assertFalse(fixture.pid_file.exists())
        self.assertEqual(fixture.events(), [])

    def test_matching_timeout_with_stale_start_ticks_is_not_signalled(self):
        fixture = self.fixture()
        supervisor = fixture.spawn([shutil.which("timeout"), "--kill-after=2s",
                                    "30s", str(fixture.backend)])
        wait_for(lambda: fixture.events(), "standalone harmless backend")
        generation = process_info(supervisor.pid)["start"]
        fixture.pid_file.write_text(f"{supervisor.pid} {generation}\n")
        self.assertEqual(fixture.running_supervisor(), supervisor.pid)
        fixture.pid_file.write_text(f"{supervisor.pid} {generation + 1}\n")
        stop = fixture.manager("stop")
        output, errors = stop.communicate(timeout=3)
        self.assertEqual(stop.returncode, 0, output + errors)
        self.assertIsNone(supervisor.poll(), "Stale process generation was signalled")
        self.assertFalse(fixture.pid_file.exists())
        self.assertIsNotNone(process_info(fixture.events()[0]["pid"]))


if __name__ == "__main__":
    unittest.main()
