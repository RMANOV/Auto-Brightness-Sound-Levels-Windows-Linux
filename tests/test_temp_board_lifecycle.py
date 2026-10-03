from contextlib import ExitStack
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location('ab_monitor', Path(__file__).resolve().parents[1] / 'scripts/temp_board_monitor.py')
tb = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tb
spec.loader.exec_module(tb)


class Cancel(KeyboardInterrupt):
    pass


class SerialTests(unittest.TestCase):
    def fake_os(self, events, opener=None):
        def default_open(path, flags):
            events.append(('open', path, flags))
            return 101
        return SimpleNamespace(O_RDWR=tb.os.O_RDWR, O_NOCTTY=tb.os.O_NOCTTY,
                               O_NONBLOCK=tb.os.O_NONBLOCK, open=opener or default_open,
                               close=lambda fd: events.append(('close', fd)))

    def fake_config(self, stage=None, error=None):
        calls = []
        def tick(name):
            calls.append(name)
            if name == stage:
                raise error
        def get(fd):
            tick('get' + str(calls.count('get1') + 1))
            return [0, 0, 0, 0, 0, 0, [0] * 32]
        term = SimpleNamespace(B115200=1, CLOCAL=1, CREAD=2, CSTOPB=4,
                               PARENB=8, CSIZE=16, HUPCL=32, CS8=64,
                               VMIN=0, VTIME=1, TCSANOW=0,
                               tcgetattr=get, tcsetattr=lambda *args: tick('set'))
        tty = SimpleNamespace(setraw=lambda fd: tick('raw'))
        return term, tty, calls

    def test_configure_failures_close_owned_descriptor_red(self):
        for stage in ['get1', 'raw', 'get2', 'set']:
            with self.subTest(stage=stage):
                events = []
                error = OSError('synthetic configure failure')
                term, tty, calls = self.fake_config(stage, error)
                with patch.object(tb, 'os', self.fake_os(events)), patch.object(tb, 'termios', term), patch.object(tb, 'tty', tty):
                    with self.assertRaises(OSError) as caught:
                        tb.open_serial('/work/synthetic/a', 115200)
                self.assertIs(caught.exception, error)
                self.assertIn(stage, calls)
                self.assertEqual([event for event in events if event[0] == 'close'], [('close', 101)])

    def test_configure_cancellation_closes_owned_descriptor_red(self):
        events = []
        error = Cancel('configure cancellation')
        term, tty, _ = self.fake_config('raw', error)
        with patch.object(tb, 'os', self.fake_os(events)), patch.object(tb, 'termios', term), patch.object(tb, 'tty', tty):
            with self.assertRaises(Cancel) as caught:
                tb.open_serial('/work/synthetic/a', 115200)
        self.assertIs(caught.exception, error)
        self.assertEqual([event for event in events if event[0] == 'close'], [('close', 101)])

    def test_unsupported_baud_closes_owned_descriptor_red(self):
        events = []
        term, tty, _ = self.fake_config()
        with patch.object(tb, 'os', self.fake_os(events)), patch.object(tb, 'termios', term), patch.object(tb, 'tty', tty):
            with self.assertRaises(ValueError):
                tb.open_serial('/work/synthetic/a', 1234567)
        self.assertEqual([event for event in events if event[0] == 'close'], [('close', 101)])

    def test_open_failure_acquires_nothing_control(self):
        events = []
        def fail(*args):
            raise OSError('open failure')
        with patch.object(tb, 'os', self.fake_os(events, fail)), patch.object(tb, 'configure_serial') as configure:
            with self.assertRaises(OSError):
                tb.open_serial('/work/synthetic/a', 115200)
            configure.assert_not_called()
        self.assertEqual(events, [])

    def test_success_transfers_descriptor_control(self):
        events = []
        term, tty, calls = self.fake_config()
        with patch.object(tb, 'os', self.fake_os(events)), patch.object(tb, 'termios', term), patch.object(tb, 'tty', tty):
            self.assertEqual(tb.open_serial('/work/synthetic/a', 115200), 101)
        self.assertEqual(calls, ['get1', 'raw', 'get2', 'set'])
        self.assertEqual(events, [('open', '/work/synthetic/a', tb.os.O_RDWR | tb.os.O_NOCTTY | tb.os.O_NONBLOCK)])

    def candidate(self, name='a'):
        return tb.SerialCandidate('/work/synthetic/' + name, '/work/synthetic/' + name,
                                  None, None, False, False, True, True, 'synthetic', 'synthetic', '-rw-------')

    def invoke(self, events, discovery, opener, writer, sleeper, extra=()):
        args = tb.parse_args(['--quiet', '--message', 'SYNTHETIC', '--open-delay', '0', *extra])
        with ExitStack() as stack:
            stack.enter_context(patch.object(tb, 'os', self.fake_os(events)))
            stack.enter_context(patch.object(tb, 'time', SimpleNamespace(sleep=sleeper)))
            stack.enter_context(patch.object(tb, 'read_host_temperatures', return_value=[]))
            stack.enter_context(patch.object(tb, 'discover_serial_candidates', side_effect=discovery))
            stack.enter_context(patch.object(tb, 'open_serial', side_effect=opener))
            stack.enter_context(patch.object(tb, 'write_serial_fd', side_effect=writer))
            return tb.monitor(args)

    def test_loop_sleep_cancellation_closes_owned_descriptor_red(self):
        events = []
        error = Cancel('loop cancellation')
        def stop(delay):
            raise error
        with self.assertRaises(Cancel) as caught:
            self.invoke(events, lambda: [self.candidate()], lambda *a: 101,
                        lambda fd, payload: events.append(('write', fd, payload)) or len(payload), stop)
        self.assertIs(caught.exception, error)
        self.assertEqual(events, [('write', 101, 'SYNTHETIC\r\n'), ('close', 101)])

    def test_loop_write_cancellation_closes_owned_descriptor_red(self):
        events = []
        error = Cancel('write cancellation')
        def write(fd, payload):
            events.append(('write', fd, payload))
            raise error
        with self.assertRaises(Cancel) as caught:
            self.invoke(events, lambda: [self.candidate()], lambda *a: 101, write,
                        lambda delay: self.fail('sleep after cancelled write'))
        self.assertIs(caught.exception, error)
        self.assertEqual(events, [('write', 101, 'SYNTHETIC\r\n'), ('close', 101)])

    def test_failed_replacement_never_closes_retired_descriptor_twice_red(self):
        events = []
        discovery = iter([[self.candidate('a')], [self.candidate('b')]])
        sleeps = []
        def open_(path, baud):
            events.append(('open', path))
            if path.endswith('/b'):
                raise OSError('replacement open failed')
            return 101
        def sleep(delay):
            sleeps.append(delay)
            if len(sleeps) == 2:
                raise Cancel('bounded stop')
        with self.assertRaises(Cancel):
            self.invoke(events, lambda: next(discovery), open_, lambda fd, payload: len(payload), sleep)
        self.assertEqual(events, [('open', '/work/synthetic/a'), ('close', 101), ('open', '/work/synthetic/b')])

    def test_replacement_closes_old_then_final_owned_descriptor_red(self):
        events = []
        discovery = iter([[self.candidate('a')], [self.candidate('b')]])
        sleeps = []
        def open_(path, baud):
            events.append(('open', path))
            return 101 if path.endswith('/a') else 202
        def sleep(delay):
            sleeps.append(delay)
            if len(sleeps) == 2:
                raise Cancel('bounded stop')
        with self.assertRaises(Cancel):
            self.invoke(events, lambda: next(discovery), open_, lambda fd, payload: len(payload), sleep)
        self.assertEqual(events, [('open', '/work/synthetic/a'), ('close', 101), ('open', '/work/synthetic/b'), ('close', 202)])

    def test_once_closes_exactly_once_control(self):
        events = []
        self.assertEqual(self.invoke(events, lambda: [self.candidate()], lambda *a: 101,
                                     lambda fd, payload: len(payload), lambda delay: self.fail('sleep forbidden'), ['--once']), 0)
        self.assertEqual(events, [('close', 101)])

    def test_dry_run_and_detect_only_never_open_control(self):
        for flag in ['--dry-run', '--detect-only']:
            events = []
            def forbidden(*args):
                self.fail('serial operation forbidden')
            self.assertEqual(self.invoke(events, lambda: [self.candidate()], forbidden, forbidden,
                                         forbidden, ['--once', flag]), 0)
            self.assertEqual(events, [])

    def test_configure_cleanup_error_preserves_primary_error_and_cancel(self):
        for error in [OSError('primary configure error'), Cancel('primary configure cancellation')]:
            with self.subTest(error=type(error).__name__):
                events = []
                fake_os = self.fake_os(events)
                def close(fd):
                    events.append(('close', fd))
                    raise OSError('cleanup failed')
                fake_os.close = close
                term, tty, _ = self.fake_config('raw', error)
                with patch.object(tb, 'os', fake_os), patch.object(tb, 'termios', term), patch.object(tb, 'tty', tty):
                    with self.assertRaises(type(error)) as caught:
                        tb.open_serial('/work/synthetic/a', 115200)
                self.assertIs(caught.exception, error)
                self.assertEqual([event for event in events if event[0] == 'close'], [('close', 101)])

    def test_loop_cleanup_error_preserves_primary_cancel(self):
        events = []
        error = Cancel('primary loop cancellation')
        fake_os = self.fake_os(events)
        def close(fd):
            events.append(('close', fd))
            raise OSError('cleanup failed')
        fake_os.close = close
        def stop(delay):
            raise error
        with patch.object(self, 'fake_os', return_value=fake_os):
            with self.assertRaises(Cancel) as caught:
                self.invoke(events, lambda: [self.candidate()], lambda *a: 101,
                            lambda fd, payload: len(payload), stop)
        self.assertIs(caught.exception, error)
        self.assertEqual(events, [('close', 101)])

    def test_existing_value_error_return2_closes_current_descriptor(self):
        events = []
        def write(fd, payload):
            raise ValueError('synthetic existing error path')
        self.assertEqual(self.invoke(events, lambda: [self.candidate()], lambda *a: 101, write,
                                     lambda delay: self.fail('sleep forbidden')), 2)
        self.assertEqual(events, [('close', 101)])

    def test_cleanup_failure_on_normal_return_does_not_report_success(self):
        events = []
        fake_os = self.fake_os(events)
        def close(fd):
            events.append(('close', fd))
            raise OSError('cleanup failed')
        fake_os.close = close
        # First iteration acquires a persistent fd; second detects only and
        # requests a normal return0, exercising the outer owned finalizer.
        args = tb.parse_args(['--quiet', '--message', 'SYNTHETIC', '--open-delay', '0'])
        def sleep(delay):
            args.detect_only = True
        with ExitStack() as stack:
            stack.enter_context(patch.object(tb, 'os', fake_os))
            stack.enter_context(patch.object(tb, 'time', SimpleNamespace(sleep=sleep)))
            stack.enter_context(patch.object(tb, 'read_host_temperatures', return_value=[]))
            stack.enter_context(patch.object(tb, 'discover_serial_candidates', return_value=[self.candidate()]))
            stack.enter_context(patch.object(tb, 'open_serial', return_value=101))
            stack.enter_context(patch.object(tb, 'write_serial_fd', return_value=11))
            with self.assertRaises(OSError):
                tb.monitor(args)
        self.assertEqual(events, [('close', 101)])

    def test_same_path_reuses_descriptor_until_cancellation(self):
        events = []
        sleeps = []
        def open_(path, baud):
            events.append(('open', path))
            return 101
        def write(fd, payload):
            events.append(('write', fd))
            return len(payload)
        def sleep(delay):
            sleeps.append(delay)
            if len(sleeps) == 2:
                raise Cancel('bounded stop')
        with self.assertRaises(Cancel):
            self.invoke(events, lambda: [self.candidate()], open_, write, sleep)
        self.assertEqual(events, [('open', '/work/synthetic/a'), ('write', 101), ('write', 101), ('close', 101)])

    def test_no_candidate_cancellation_owns_no_descriptor(self):
        events = []
        def forbidden(*args):
            self.fail('serial operation forbidden')
        def stop(delay):
            raise Cancel('bounded stop')
        with self.assertRaises(Cancel):
            self.invoke(events, lambda: [], forbidden, forbidden, stop)
        self.assertEqual(events, [])
