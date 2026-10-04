"""Real firmware source contracts, with hardware imports denied before access."""
import builtins
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


class RecordingOLED:
    def __init__(self):
        self.calls = []
    def fill(self, value):
        self.calls.append(('fill', value))
    def text(self, text, x, y):
        self.calls.append(('text', text, x, y))
    def show(self):
        self.calls.append(('show',))
    def vline(self, x, y, height, color):
        self.calls.append(('vline', x, y, height, color))
    def rect(self, x, y, width, height, color):
        self.calls.append(('rect', x, y, width, height, color))
    def fill_rect(self, x, y, width, height, color):
        self.calls.append(('fill_rect', x, y, width, height, color))


class FirmwareTests(unittest.TestCase):
    def load(self):
        source = Path(__file__).resolve().parents[1] / 'scripts/temp_board_firmware/main.py'
        spec = importlib.util.spec_from_file_location('ab_firmware_synthetic', source)
        module = importlib.util.module_from_spec(spec)
        attempted = []
        def closed_import(name, *args, **kwargs):
            if name.split('.')[0] in {'machine', 'ssd1306', 'micropython', 'framebuf'}:
                attempted.append(name)
                raise ImportError('owned hardware import prohibited: ' + name)
            return builtins.__import__(name, *args, **kwargs)
        # This is module-local: never replace process-global builtins/import state.
        module.__dict__['__builtins__'] = dict(vars(builtins), __import__=closed_import)
        error = None
        try:
            spec.loader.exec_module(module)
        except ImportError as caught:
            error = caught
        self.assertIsNone(error, 'firmware import must not request hardware: ' + repr(error))
        self.assertEqual(attempted, [])
        return module

    def test_import_without_hardware(self):
        module = self.load()
        self.assertTrue(callable(module.parse_key_payload))
        self.assertTrue(callable(module.parse_legacy_payload))
        self.assertFalse(hasattr(module, 'oled'))
        self.assertFalse(hasattr(module, 'i2c'))
        self.assertFalse(hasattr(module, 'uart'))

    def test_valid_key_wire(self):
        module = self.load()
        self.assertEqual(module.parse_key_payload('CPU=42|PCH=55|NVME=60|LOAD=25|RAM=35|BAT=80|AC=1|FAN=1200'),
                         {'cpu': 42, 'pch': 55, 'nvme': 60, 'load': 25, 'ram': 35, 'bat': 80, 'ac': '1', 'fan': 1200})

    def test_missing_cpu(self):
        module = self.load()
        for line in ['', 'PCH=20|NVME=30', 'garbage', 'CPU42']:
            with self.subTest(line=line):
                self.assertIsNone(module.parse_key_payload(line))

    def test_malformed_nonfinite(self):
        module = self.load()
        self.assertEqual(module.parse_key_payload('CPU=nan|PCH=bad|NVME=inf|BAT=nan|FAN=oops'),
                         {'cpu': 0, 'pch': 0, 'nvme': 0, 'load': 0, 'ram': 0, 'bat': 100, 'ac': '1', 'fan': 0})
        for value in ['nan', 'inf', '-inf', '', 'bad']:
            with self.subTest(value=value):
                self.assertEqual(module.clamp_int(value, default=7), 7)

    def test_partial_defaults(self):
        module = self.load()
        self.assertEqual(module.parse_key_payload('CPU=12'),
                         {'cpu': 12, 'pch': 0, 'nvme': 0, 'load': 0, 'ram': 0, 'bat': 100, 'ac': '1', 'fan': 0})

    def test_duplicate_case_unknown(self):
        module = self.load()
        self.assertEqual(module.parse_key_payload(' cpu = 12 |CPU=34|X=9'),
                         {'cpu': 34, 'pch': 0, 'nvme': 0, 'load': 0, 'ram': 0, 'bat': 100, 'ac': '1', 'fan': 0})

    def test_legacy_exact_seven(self):
        module = self.load()
        # Existing CPU/load conflation is characterized, not certified as temperature.
        self.assertEqual(module.parse_legacy_payload('CPU12C|GPU--|25|35|0|80|1'),
                         {'cpu': 25, 'pch': 0, 'nvme': 0, 'load': 25, 'ram': 35, 'bat': 80, 'ac': '1', 'fan': 0})

    def test_legacy_bad_arity(self):
        module = self.load()
        for line in ['', 'a|b|c|d|e|f', 'a|b|c|d|e|f|g|h']:
            with self.subTest(line=line):
                self.assertIsNone(module.parse_legacy_payload(line))

    def test_clamp_boundaries(self):
        module = self.load()
        self.assertEqual(module.parse_key_payload('CPU=-5|PCH=999|LOAD=150|RAM=-2|BAT=500|FAN=20000|AC=true'),
                         {'cpu': 0, 'pch': 120, 'nvme': 0, 'load': 100, 'ram': 0, 'bat': 100, 'ac': '0', 'fan': 9999})
        for value, expected in [('0', 0), ('120', 120), ('119.9', 119), ('-1', 0), ('121', 120)]:
            with self.subTest(value=value):
                self.assertEqual(module.clamp_int(value), expected)

    def test_draw_cpu_pch_nv(self):
        module = self.load()
        oled = RecordingOLED()
        with patch.object(module, 'oled', oled, create=True):
            module.draw_status({'cpu': 0, 'pch': 0, 'nvme': 0})
        self.assertEqual(oled.calls[:3], [('fill', 0), ('vline', 42, 0, 64, 1), ('vline', 85, 0, 64, 1)])
        texts = [call for call in oled.calls if call[0] == 'text']
        self.assertEqual(texts, [('text', 'CPU', 9, 2), ('text', 'C', 33, 52), ('text', 'PCH', 52, 2),
                                 ('text', 'C', 76, 52), ('text', 'NV', 99, 2), ('text', 'C', 119, 52)])
        # Three zero digits, each drawn by the six non-middle segments.
        self.assertEqual(sum(call[0] == 'fill_rect' for call in oled.calls), 18)
        self.assertEqual(oled.calls[-1], ('show',))

    def test_bar_clamp(self):
        module = self.load()
        for value, expected in [(-5, [('rect', 2, 3, 10, 7, 1)]),
                                (75, [('rect', 2, 3, 10, 7, 1), ('fill_rect', 3, 4, 6, 5, 1)]),
                                (150, [('rect', 2, 3, 10, 7, 1), ('fill_rect', 3, 4, 8, 5, 1)])]:
            oled = RecordingOLED()
            with self.subTest(value=value), patch.object(module, 'oled', oled, create=True):
                module.bar(2, 3, 10, value)
                self.assertEqual(oled.calls, expected)

    def test_unknown_digit(self):
        module = self.load()
        oled = RecordingOLED()
        with patch.object(module, 'oled', oled, create=True):
            module.segment_digit(0, 0, '?')
        self.assertEqual(oled.calls, [])

    def test_clear_message_truncates(self):
        module = self.load()
        oled = RecordingOLED()
        with patch.object(module, 'oled', oled, create=True):
            module.clear_msg('abcdefghijklmnopQ', '1234567890123456X', '')
        self.assertEqual(oled.calls, [('fill', 0), ('text', 'abcdefghijklmnop', 0, 8),
                                      ('text', '1234567890123456', 0, 24), ('show',)])
