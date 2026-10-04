from contextlib import ExitStack
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('ab_reliability', Path(__file__).resolve().parents[1] / 'scripts/temp_board_monitor.py')
tb = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tb
spec.loader.exec_module(tb)


class Cancel(KeyboardInterrupt):
    pass


class ReliabilityTests(unittest.TestCase):
    def candidate(self, writable=True):
        return tb.SerialCandidate('/work/synthetic/a', '/work/synthetic/a', None, None,
                                  False, False, True, writable, 'synthetic', 'synthetic', '-rw-------')

    def invoke(self, candidates=None, flags=(), write_error=None, read_error=None, reading='0'):
        args = tb.parse_args(['--once', '--quiet', '--message', 'SYNTHETIC', '--open-delay', '0', *flags])
        with ExitStack() as stack:
            stack.enter_context(patch.object(tb, 'read_host_temperatures', return_value=[]))
            stack.enter_context(patch.object(tb, 'discover_serial_candidates', return_value=[self.candidate()] if candidates is None else candidates))
            stack.enter_context(patch.object(tb, 'time', SimpleNamespace(sleep=lambda *_: self.fail('once must not loop'))))
            stack.enter_context(patch.object(tb, 'write_serial_once', side_effect=write_error, return_value=11))
            stack.enter_context(patch.object(tb, 'read_serial_once', side_effect=read_error, return_value=reading))
            return tb.monitor(args)

    def test_once_missing_port_fails_red(self):
        self.assertEqual(self.invoke(candidates=[]), 1)

    def test_once_inaccessible_port_fails_red(self):
        self.assertEqual(self.invoke(candidates=[self.candidate(False)]), 1)

    def test_once_write_error_fails_red(self):
        for error in [PermissionError('synthetic'), OSError('synthetic')]:
            with self.subTest(error=type(error).__name__):
                self.assertEqual(self.invoke(write_error=error), 1)

    def test_once_read_error_fails_red(self):
        for error in [PermissionError('synthetic'), OSError('synthetic')]:
            with self.subTest(error=type(error).__name__):
                self.assertEqual(self.invoke(flags=['--read-board'], read_error=error), 1)

    def test_once_no_numeric_read_fails_red(self):
        for reading in [None, '', 'synthetic nonnumeric']:
            with self.subTest(reading=reading):
                self.assertEqual(self.invoke(flags=['--read-board'], reading=reading), 1)

    def test_once_full_write_and_zero_numeric_read_succeed_control(self):
        self.assertEqual(self.invoke(), 0)
        self.assertEqual(self.invoke(flags=['--read-board'], reading='0'), 0)

    def test_configuration_and_usage_status2_control(self):
        self.assertEqual(self.invoke(write_error=ValueError('baud')), 2)
        self.assertEqual(self.invoke(flags=['--read-board'], read_error=ValueError('baud')), 2)
        self.assertEqual(tb.main(['--interval', '0']), 2)

    def test_diagnostics_missing_inaccessible_ports_status0_control(self):
        for flag in ['--dry-run', '--detect-only']:
            for candidates in [[], [self.candidate(False)]]:
                with self.subTest(flag=flag, inaccessible=bool(candidates)):
                    self.assertEqual(self.invoke(candidates=candidates, flags=[flag],
                                                 write_error=AssertionError('diagnostic write'),
                                                 read_error=AssertionError('diagnostic read')), 0)

    def fake_io(self, close_error=None):
        events = []
        def close(fd):
            events.append(('close', fd))
            if close_error is not None:
                raise close_error
        fake = SimpleNamespace(O_RDWR=tb.os.O_RDWR, O_NOCTTY=tb.os.O_NOCTTY,
                               O_NONBLOCK=tb.os.O_NONBLOCK, open=lambda *a: 101, close=close)
        return fake, events

    def test_recurring_operational_errors_reach_retry_sleep_control(self):
        for read in [False, True]:
            args = tb.parse_args(['--quiet', '--message', 'SYNTHETIC', '--open-delay', '0', *(['--read-board'] if read else [])])
            stop = Cancel('bounded retry stop')
            with self.subTest(read=read), ExitStack() as stack:
                stack.enter_context(patch.object(tb, 'read_host_temperatures', return_value=[]))
                stack.enter_context(patch.object(tb, 'discover_serial_candidates', return_value=[self.candidate()]))
                stack.enter_context(patch.object(tb, 'open_serial', return_value=101))
                stack.enter_context(patch.object(tb, 'write_serial_fd', side_effect=OSError('write')))
                stack.enter_context(patch.object(tb, 'read_serial_once', side_effect=OSError('read')))
                stack.enter_context(patch.object(tb, 'os', SimpleNamespace(close=lambda _: None)))
                stack.enter_context(patch.object(tb, 'time', SimpleNamespace(sleep=lambda _: (_ for _ in ()).throw(stop))))
                with self.assertRaises(Cancel) as caught:
                    tb.monitor(args)
                self.assertIs(caught.exception, stop)

    def capture(self, operation):
        try:
            operation()
        except BaseException as error:
            return error
        self.fail('expected exception')

    def test_owned_write_primary_identity_survives_close_error_red(self):
        for primary in [OSError('primary write'), Cancel('primary write cancel')]:
            fake, events = self.fake_io(OSError('cleanup'))
            with self.subTest(primary=type(primary).__name__), patch.object(tb, 'os', fake), patch.object(tb, 'open_serial', return_value=101), patch.object(tb, 'write_serial_fd', side_effect=primary):
                actual = self.capture(lambda: tb.write_serial_once('/work/synthetic/a', 115200, 'SYNTHETIC', 0))
            self.assertIs(actual, primary)
            self.assertEqual(events, [('close', 101)])

    def test_owned_write_sleep_cancel_survives_close_error_red(self):
        primary = Cancel('primary delay cancel')
        fake, events = self.fake_io(OSError('cleanup'))
        with patch.object(tb, 'os', fake), patch.object(tb, 'open_serial', return_value=101), patch.object(tb, 'time', SimpleNamespace(sleep=lambda _: (_ for _ in ()).throw(primary))):
            actual = self.capture(lambda: tb.write_serial_once('/work/synthetic/a', 115200, 'SYNTHETIC', 1))
        self.assertIs(actual, primary)
        self.assertEqual(events, [('close', 101)])

    def test_owned_read_config_primary_survives_close_error_red(self):
        for primary in [OSError('primary configure'), Cancel('primary configure cancel')]:
            fake, events = self.fake_io(OSError('cleanup'))
            with self.subTest(primary=type(primary).__name__), patch.object(tb, 'os', fake), patch.object(tb, 'configure_serial', side_effect=primary):
                actual = self.capture(lambda: tb.read_serial_once('/work/synthetic/a', 115200))
            self.assertIs(actual, primary)
            self.assertEqual(events, [('close', 101)])

    def test_owned_read_io_primary_survives_close_error_red(self):
        for primary in [OSError('primary read'), Cancel('primary read cancel')]:
            fake, events = self.fake_io(OSError('cleanup'))
            fake.read = lambda *a: (_ for _ in ()).throw(primary)
            with self.subTest(primary=type(primary).__name__), patch.object(tb, 'os', fake), patch.object(tb, 'configure_serial'), patch.object(tb, 'time', SimpleNamespace(monotonic=lambda: 0)), patch.object(tb, 'select', SimpleNamespace(select=lambda *a: ([101], [], []))):
                actual = self.capture(lambda: tb.read_serial_once('/work/synthetic/a', 115200))
            self.assertIs(actual, primary)
            self.assertEqual(events, [('close', 101)])

    def test_owned_normal_returns_do_not_hide_close_error_control(self):
        cleanup = OSError('cleanup')
        fake, events = self.fake_io(cleanup)
        with patch.object(tb, 'os', fake), patch.object(tb, 'open_serial', return_value=101), patch.object(tb, 'write_serial_fd', return_value=11):
            self.assertIs(self.capture(lambda: tb.write_serial_once('/work/synthetic/a', 115200, 'SYNTHETIC', 0)), cleanup)
        self.assertEqual(events, [('close', 101)])
        fake, events = self.fake_io(cleanup)
        with patch.object(tb, 'os', fake), patch.object(tb, 'configure_serial'), patch.object(tb, 'time', SimpleNamespace(monotonic=lambda: 0)):
            self.assertIs(self.capture(lambda: tb.read_serial_once('/work/synthetic/a', 115200, timeout=0)), cleanup)
        self.assertEqual(events, [('close', 101)])

    def configure(self, with_crtscts):
        calls = []
        flag = 128
        unrelated = 256
        term = SimpleNamespace(B115200=1, CLOCAL=1, CREAD=2, CSTOPB=4,
                               PARENB=8, CSIZE=16, HUPCL=32, CS8=64,
                               VMIN=0, VTIME=1, TCSANOW=0)
        if with_crtscts:
            term.CRTSCTS = flag
        def get(fd):
            calls.append('get')
            return [7, 9, flag | unrelated | 4 | 8 | 16 | 32, 11, 0, 0, [0] * 32]
        applied = []
        term.tcgetattr = get
        term.tcsetattr = lambda fd, when, attrs: (calls.append('set'), applied.append(attrs))
        with patch.object(tb, 'termios', term), patch.object(tb, 'tty', SimpleNamespace(setraw=lambda fd: calls.append('raw'))):
            tb.configure_serial(101, 115200)
        self.assertEqual(calls, ['get', 'raw', 'get', 'set'])
        self.assertEqual(applied[0][2] & (1 | 2 | 64 | unrelated), 1 | 2 | 64 | unrelated)
        self.assertEqual(applied[0][2] & (4 | 8 | 16 | 32), 0)
        self.assertEqual(applied[0][4:6], [1, 1])
        return applied[0][2]

    def test_normal_cleanup_error_propagates_inside_unrelated_caller_exception_control(self):
        for read in [False, True]:
            cleanup = OSError('own cleanup')
            fake, events = self.fake_io(cleanup)
            with self.subTest(read=read), ExitStack() as stack:
                stack.enter_context(patch.object(tb, 'os', fake))
                stack.enter_context(patch.object(tb, 'open_serial', return_value=101))
                stack.enter_context(patch.object(tb, 'write_serial_fd', return_value=11))
                stack.enter_context(patch.object(tb, 'configure_serial'))
                stack.enter_context(patch.object(tb, 'time', SimpleNamespace(monotonic=lambda: 0)))
                try:
                    raise ValueError('unrelated caller')
                except ValueError:
                    if read:
                        actual = self.capture(lambda: tb.read_serial_once('/work/synthetic/a', 115200, timeout=0))
                    else:
                        actual = self.capture(lambda: tb.write_serial_once('/work/synthetic/a', 115200, 'SYNTHETIC', 0))
                self.assertIs(actual, cleanup)
                self.assertEqual(events, [('close', 101)])

    def test_configure_clears_optional_crtscts_red(self):
        self.assertEqual(self.configure(True) & 128, 0)

    def test_configure_without_crtscts_preserves_unrelated_flag_control(self):
        self.assertEqual(self.configure(False) & 128, 128)
