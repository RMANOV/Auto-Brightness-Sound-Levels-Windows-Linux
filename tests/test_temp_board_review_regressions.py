"""Actual sender/firmware regressions with closed IO and an advancing fake clock."""
import builtins
import fnmatch
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, SOURCE / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    def closed_import(name, *args, **kwargs):
        if name.split('.')[0] in {'machine', 'ssd1306', 'micropython', 'framebuf', 'uselect'}:
            raise AssertionError('hardware import during source load: ' + name)
        return builtins.__import__(name, *args, **kwargs)
    module.__dict__['__builtins__'] = {**vars(builtins), '__import__': closed_import}
    spec.loader.exec_module(module)
    return module


class StopLoop(BaseException):
    pass


class ReviewRegressions(unittest.TestCase):
    def sender(self):
        return load('ab_review_sender', 'scripts/temp_board_monitor.py')

    def firmware(self):
        return load('ab_review_firmware', 'scripts/temp_board_firmware/main.py')

    def write(self, script, payload='ABC\r\n'):
        module = self.sender()
        actions = iter(script)
        trace = []
        clock = SimpleNamespace(value=0.0)
        def write(fd, data):
            self.assertEqual(fd, 101)
            trace.append(('write', bytes(data)))
            action = next(actions)
            if isinstance(action, BaseException):
                raise action
            return action
        def ready(read, write, errors, delay):
            self.assertEqual((read, write, errors), ([], [101], []))
            self.assertGreater(delay, 0)
            self.assertLessEqual(delay, 2)
            trace.append(('wait', delay))
            clock.value += delay
            return [], [101], []
        with patch.object(module, 'os', SimpleNamespace(write=write)), \
             patch.object(module, 'select', SimpleNamespace(select=ready)), \
             patch.object(module, 'time', SimpleNamespace(monotonic=lambda: clock.value)), \
             patch.object(module, 'termios', SimpleNamespace(tcdrain=lambda fd: trace.append(('drain', fd)))):
            result = module.write_serial_fd(101, payload)
        return result, trace

    def test_short_write_then_backpressure_sends_exact_remainder(self):
        result, trace = self.write([2, BlockingIOError(), 3])
        self.assertEqual(result, 5)
        self.assertEqual([x[1] for x in trace if x[0] == 'write'], [b'ABC\r\n', b'C\r\n', b'C\r\n'])
        self.assertEqual(trace[-1], ('drain', 101))
        self.assertEqual(sum(x[0] == 'drain' for x in trace), 1)

    def test_zero_progress_and_backpressure_have_finite_deadline(self):
        for action in (0, BlockingIOError()):
            with self.subTest(action=type(action).__name__), self.assertRaises(TimeoutError):
                self.write([action] * 100)

    def test_interrupted_write_retries_without_losing_bytes(self):
        result, trace = self.write([InterruptedError(), 5])
        self.assertEqual(result, 5)
        self.assertEqual([x[1] for x in trace if x[0] == 'write'], [b'ABC\r\n'] * 2)

    def test_utf8_counts_bytes_and_full_write_control(self):
        result, trace = self.write([1, 4], '\N{DEGREE SIGN}C\r\n')
        self.assertEqual(result, 5)
        self.assertEqual(trace[1], ('write', b'\xb0C\r\n'))
        self.assertEqual(self.write([5])[0], 5)

    def test_permanent_write_error_and_cancellation_are_not_success(self):
        for error in (OSError('synthetic permanent failure'), StopLoop('synthetic cancellation')):
            with self.subTest(error=type(error).__name__):
                with self.assertRaises(type(error)) as caught:
                    self.write([2, error])
                self.assertIs(caught.exception, error)

    def power(self, supplies):
        module = self.sender()
        files = {}
        for name, kind, online in supplies:
            files['/sys/class/power_supply/' + name + '/type'] = kind
            if online is not None:
                files['/sys/class/power_supply/' + name + '/online'] = online
        roots = ['/sys/class/power_supply/' + name for name, _, _ in supplies]
        class FakePath:
            def __init__(self, value):
                self.value = str(value)
                if not self.value.startswith('/sys/class/power_supply'):
                    raise AssertionError('outside synthetic power root')
            def __truediv__(self, name):
                return FakePath(self.value + '/' + name)
            def __str__(self):
                return self.value
            def __lt__(self, other):
                return self.value < other.value
            def glob(self, pattern):
                if self.value != '/sys/class/power_supply':
                    raise AssertionError('unexpected power glob')
                if pattern == '*':
                    return [FakePath(value) for value in roots]
                if pattern not in ('BAT*/capacity', 'A*C*/online'):
                    raise AssertionError('unexpected power pattern: ' + pattern)
                return [FakePath(value) for value in files if fnmatch.fnmatch(value, self.value + '/' + pattern)]
        with patch.object(module, 'Path', FakePath), patch.object(module, 'read_text', lambda path: files.get(str(path))):
            return module.read_battery_state()

    def test_mains_identity_not_adapter_name(self):
        self.assertEqual(self.power([('ADP1', 'Mains', '0')]), (100, 0))
        self.assertEqual(self.power([('ADP1', 'Mains', '1')]), (100, 1))
        self.assertEqual(self.power([('AC', 'Battery', '0')]), (100, 1))

    def test_multiple_mains_online_and_unknown_fallback(self):
        self.assertEqual(self.power([('ADP1', 'Mains', '0'), ('USB1', 'Mains', '1')]), (100, 1))
        self.assertEqual(self.power([('ADP1', 'Mains', '0'), ('USB1', 'Mains', '0')]), (100, 0))
        self.assertEqual(self.power([('ADP1', 'Mains', None)]), (100, 1))

    def test_temperature_prefix_does_not_require_new_string_api(self):
        module = self.sender()
        class OldName(str):
            def __getattribute__(self, name):
                if name == 'removesuffix':
                    raise AttributeError('synthetic pre3.9 string API')
                return super().__getattribute__(name)
        class FakePath:
            def __init__(self, value):
                self.value = str(value)
                if not self.value.startswith(('/sys/class/hwmon', '/sys/class/thermal')):
                    raise AssertionError('outside synthetic thermal roots')
            def __truediv__(self, name):
                return FakePath(self.value + '/' + name)
            @property
            def name(self):
                return OldName(self.value.rsplit('/', 1)[-1])
            def glob(self, pattern):
                values = {'/sys/class/hwmon': ['hwmon0'], '/sys/class/hwmon/hwmon0': ['temp1_input'], '/sys/class/thermal': []}
                return [self / name for name in values[self.value]]
        values = {'/sys/class/hwmon/hwmon0/name': 'coretemp', '/sys/class/hwmon/hwmon0/temp1_input': '42000',
                  '/sys/class/hwmon/hwmon0/temp1_label': 'Package id 0'}
        with patch.object(module, 'Path', FakePath), patch.object(module, 'read_text', lambda path: values.get(path.value)):
            readings = module.read_host_temperatures()
        self.assertEqual([(x.label, x.celsius) for x in readings], [('coretemp:Package id 0', 42.0)])

    def test_explicit_temp_bars_header_preserves_pch(self):
        module = self.firmware()
        for header in ('NV60C', 'NVME--'):
            with self.subTest(header=header):
                data = module.parse_legacy_payload('CPU42C|' + header + '|42|55|60|80|1')
                self.assertEqual((data['cpu'], data['pch'], data['nvme']), (42, 55, 60))
                self.assertEqual((data['load'], data['ram']), (0, 0))

    def test_pc_monitor_and_key_value_controls(self):
        module = self.firmware()
        self.assertEqual(module.parse_legacy_payload('CPU12C|GPU--|25|35|0|80|1'),
                         {'cpu': 12, 'pch': 0, 'nvme': 0, 'load': 25, 'ram': 35, 'bat': 80, 'ac': '1', 'fan': 0})
        data = module.parse_key_payload('CPU=42|PCH=55|NVME=60')
        self.assertEqual((data['cpu'], data['pch'], data['nvme']), (42, 55, 60))

    def test_pc_monitor_actual_producer_temperature_is_not_cpu_load(self):
        sender = self.sender()
        with patch.object(sender, 'read_cpu_percent', return_value=25), \
             patch.object(sender, 'read_memory_percent', return_value=35), \
             patch.object(sender, 'read_battery_state', return_value=(80, 1)):
            payload = sender.format_pc_monitor_payload(sender.Temperature('synthetic', 'cpu', 12.0))
        self.assertEqual(payload, 'CPU12C|GPU--|25|35|0|80|1')
        data = self.firmware().parse_legacy_payload(payload, strict=True)
        self.assertEqual((data['cpu'], data['load'], data['ram']), (12, 25, 35))

    def test_celsius_region_does_not_overwrite_digit_pixels(self):
        module = self.firmware()
        for x0 in (0, 43, 86):
            for value in (0, 99, 100, 120):
                with self.subTest(x0=x0, value=value):
                    pixels, units = set(), []
                    def rect(x, y, width, height, color):
                        pixels.update((a, b) for a in range(x, x + width) for b in range(y, y + height))
                    def text(value, x, y):
                        if value == 'C':
                            units.append((x, y))
                    with patch.object(module, 'oled', SimpleNamespace(fill_rect=rect, text=text), create=True):
                        module.section(x0, 42, 'CPU', value)
                    self.assertEqual(units, [(x0 + 33, 52)])
                    unit = {(a, b) for a in range(x0 + 33, x0 + 41) for b in range(52, 60)}
                    self.assertFalse(pixels & unit, 'unit collides with a digit segment')
                    self.assertTrue(all(x0 <= x < x0 + 42 and 20 <= y < 56 for x, y in pixels))

    def run_receiver(self, chunks, stop_at, start=0):
        module = self.firmware()
        clock = SimpleNamespace(elapsed=0)
        draws, notices = [], []
        pending = list(chunks)
        stream_buffer = []
        class Stream:
            def readline(self):
                raise StopLoop('blocking readline reached')
            def read(self, count):
                self.assert_one(count)
                if not stream_buffer:
                    raise AssertionError('unready read')
                return stream_buffer.pop(0)
            def assert_one(self, count):
                if count != 1:
                    raise AssertionError('partial line could block')
        stream = Stream()
        class Poll:
            def register(self, actual, flags):
                if actual is not stream or flags != 1:
                    raise AssertionError('wrong stream registration')
            def poll(self, delay):
                if delay != 0:
                    raise AssertionError('poll must not block')
                while pending and pending[0][0] <= clock.elapsed:
                    _, text = pending.pop(0)
                    stream_buffer.extend(text)
                return [(stream, 1)] if stream_buffer else []
        def pause(milliseconds):
            if milliseconds != 50:
                raise AssertionError('unexpected polling cadence')
            clock.elapsed += milliseconds
            if clock.elapsed >= stop_at:
                raise StopLoop('bounded synthetic end')
        fake_time = SimpleNamespace(ticks_ms=lambda: (start + clock.elapsed) % 16384,
            ticks_diff=lambda a, b: (a - b + 8192) % 16384 - 8192,
            sleep_ms=pause, sleep=lambda seconds: pause(int(seconds * 1000)))
        fake_modules = {'machine': SimpleNamespace(I2C=lambda **kwargs: None, Pin=lambda pin: pin, UART=lambda *args, **kwargs: None),
                        'ssd1306': SimpleNamespace(SSD1306_I2C=lambda *args: SimpleNamespace()),
                        'uselect': SimpleNamespace(poll=Poll, POLLIN=1)}
        def imports(name, *args, **kwargs):
            if name not in fake_modules:
                raise AssertionError('undeclared runtime import: ' + name)
            return fake_modules[name]
        module.__dict__['__builtins__']['__import__'] = imports
        with patch.object(module, 'sys', SimpleNamespace(stdin=stream)), patch.object(module, 'time', fake_time), \
             patch.object(module, 'clear_msg', lambda *args: notices.append((clock.elapsed, args))), \
             patch.object(module, 'draw_status', lambda data: draws.append((clock.elapsed, data.copy()))):
            with self.assertRaises(StopLoop):
                module.main()
        return draws, notices

    def test_receiver_valid_payload_expires_exactly_and_recovers(self):
        draws, notices = self.run_receiver([(0, 'CPU=42|PCH=55\n'), (6500, 'CPU=43\n')], 7000)
        self.assertEqual([(t, x['cpu']) for t, x in draws], [(0, 42), (6500, 43)])
        self.assertEqual([t for t, x in notices if x[:2] == ('THERMAL WATCH', 'waiting serial')], [0, 5000])

    def test_receiver_partial_bad_empty_do_not_refresh_valid_time(self):
        draws, notices = self.run_receiver([(0, 'CPU=42\n'), (4900, 'garbage\n\nCPU=')], 5500)
        self.assertEqual([(t, x['cpu']) for t, x in draws], [(0, 42)])
        self.assertIn((5000, ('THERMAL WATCH', 'waiting serial', '115200 baud')), notices)

    def test_receiver_ticks_wrap_and_never_reads_partial_line_blockingly(self):
        draws, notices = self.run_receiver([(0, 'CPU='), (100, '42\n')], 5250, start=16000)
        self.assertEqual([(t, x['cpu']) for t, x in draws], [(100, 42)])
        self.assertIn((5100, ('THERMAL WATCH', 'waiting serial', '115200 baud')), notices)

    def test_receiver_bad_numeric_temperatures_do_not_refresh_or_draw(self):
        for bad in ('CPU=bad', 'CPU=nan', 'CPU=inf', 'CPU=', 'CPU=42|PCH=bad',
                    'CPU=42|NVME=-inf', 'PCH=55', 'X=1',
                    'CPU42C|NV60C|42|bad|60|80|1'):
            with self.subTest(payload=bad):
                draws, notices = self.run_receiver([(0, 'CPU=42\n'), (4900, bad + '\n')], 5500)
                self.assertEqual([(t, x['cpu']) for t, x in draws], [(0, 42)])
                self.assertIn((5000, ('THERMAL WATCH', 'waiting serial', '115200 baud')), notices)

    def test_receiver_valid_partial_temperature_payload_refreshes(self):
        draws, notices = self.run_receiver([(0, 'CPU=42\n'), (4900, 'CPU=43\n')], 5500)
        self.assertEqual([(t, x['cpu'], x['pch']) for t, x in draws], [(0, 42, 0), (4900, 43, 0)])
        self.assertEqual([t for t, x in notices if x[:2] == ('THERMAL WATCH', 'waiting serial')], [0])

    def test_receiver_unknown_or_bad_temperature_header_does_not_refresh(self):
        for header in ('CPUNAC', 'CPUbadC', 'CPUnanC', 'CPUinfC', 'CPUC'):
            for other in ('GPU--', 'NV60C'):
                with self.subTest(header=header, other=other):
                    payload = header + '|' + other + '|25|35|0|80|1\n'
                    draws, notices = self.run_receiver([(0, 'CPU=42\n'), (4900, payload)], 5500)
                    self.assertEqual([(t, x['cpu']) for t, x in draws], [(0, 42)])
                    self.assertIn((5000, ('THERMAL WATCH', 'waiting serial', '115200 baud')), notices)

    def test_receiver_oversize_line_discard_and_resynchronization(self):
        draws, _ = self.run_receiver([(0, 'x' * 300 + 'CPU=99\nCPU=42\n')], 500)
        self.assertEqual([x['cpu'] for _, x in draws], [42])


if __name__ == '__main__':
    unittest.main()
