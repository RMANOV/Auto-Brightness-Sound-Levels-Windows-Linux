"""Synthetic Linux telemetry contracts; never read host proc/sysfs or hardware."""
from contextlib import ExitStack
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('ab_cpu_telemetry', Path(__file__).resolve().parents[1] / 'scripts/temp_board_monitor.py')
tb = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tb
spec.loader.exec_module(tb)


class ClosedFS:
    """Only two synthetic ABI roots; unspecified metadata is an absent file."""
    def __init__(self, hwmon=(), thermal=(), aliases=None):
        self.files = {}
        self.groups = {'/sys/class/hwmon': [], '/sys/class/thermal': []}
        self.aliases = aliases or {}
        for index, (driver, sensors) in enumerate(hwmon):
            root = f'/sys/class/hwmon/hwmon{index}'
            self.groups['/sys/class/hwmon'].append(root)
            self.files[root + '/name'] = driver
            self.groups[root] = []
            for slot, label, value in sensors:
                name = root + f'/temp{slot}_input'
                self.groups[root].append(name)
                self.files[name] = value
                self.files[root + f'/temp{slot}_label'] = label
        for index, (kind, value) in enumerate(thermal):
            root = f'/sys/class/thermal/thermal_zone{index}'
            self.groups['/sys/class/thermal'].append(root)
            self.files[root + '/type'] = kind
            self.files[root + '/temp'] = value
        fs = self
        class FakePath:
            def __init__(self, value):
                self.value = str(value)
                if not self.value.startswith(('/sys/class/hwmon', '/sys/class/thermal', '/synthetic/')):
                    raise AssertionError('outside closed filesystem: ' + self.value)
            def __str__(self):
                return self.value
            def __truediv__(self, suffix):
                return FakePath(self.value + '/' + suffix)
            @property
            def name(self):
                return self.value.rsplit('/', 1)[-1]
            def glob(self, pattern):
                allowed = {'/sys/class/hwmon': 'hwmon*', '/sys/class/thermal': 'thermal_zone*'}
                if pattern != allowed.get(self.value, 'temp*_input'):
                    raise AssertionError('unexpected glob: ' + pattern)
                return [FakePath(p) for p in fs.groups.get(self.value, [])]
            def resolve(self):
                return FakePath(fs.aliases.get(self.value, self.value))
            def __lt__(self, other):
                return self.value < other.value
            def __hash__(self):
                return hash(self.value)
            def __eq__(self, other):
                return isinstance(other, FakePath) and self.value == other.value
        self.Path = FakePath
    def read(self, path):
        value = str(path)
        if not value.startswith(('/sys/class/hwmon/', '/sys/class/thermal/')):
            raise AssertionError('outside closed read: ' + value)
        return self.files.get(value)
    def collect(self):
        with patch.object(tb, 'Path', self.Path), patch.object(tb, 'read_text', self.read):
            return tb.read_host_temperatures()


class CpuTelemetryTests(unittest.TestCase):
    def T(self, source, label, value):
        return tb.Temperature(source, label, value)
    def totals(self, raw):
        with patch.object(tb, 'read_text', lambda path: raw if str(path) == '/proc/stat' else self.fail('unexpected proc path')):
            try:
                return tb.read_proc_stat_cpu_totals()
            except (IndexError, ValueError) as error:
                return error  # Invalid input is a genuine assertion failure, not a fixture ERROR.
    def test_guest_totals_exclude_duplicate_fields(self):
        for tail in ['', ' 900 700']:
            with self.subTest(tail=tail):
                self.assertEqual(self.totals('cpu 100 20 30 400 10 5 6 7 50 10' + tail), (410, 578))
    def test_guest_delta_percent(self):
        snapshots = iter(['cpu 100 0 0 100 0 0 0 0 80 0', 'cpu 150 0 0 150 0 0 0 0 130 0'])
        with patch.object(tb, 'read_text', lambda path: next(snapshots) if str(path) == '/proc/stat' else self.fail('unexpected path')), patch.object(tb, 'time', SimpleNamespace(sleep=lambda delay: None)):
            self.assertEqual(tb.read_cpu_percent(), 50)
    def test_short_negative_stat_is_unavailable(self):
        for raw in ['', 'cpu', 'cpu 1 2 3', 'cpu 1 2 nope 4', 'cpu -1 2 3 4', 'cpu 1 2 3 4 -1']:
            with self.subTest(raw=raw):
                self.assertIsNone(self.totals(raw))
    def test_older_stat_and_delta_controls(self):
        for raw, expected in [('cpu 1 2 3 4', (4, 10)), ('cpu 1 2 3 4 5 6 7 8', (9, 36)), ('cpu0 1 2 3 4', None), (None, None)]:
            with self.subTest(raw=raw):
                self.assertEqual(self.totals(raw), expected)
        for pair, expected in [([(10, 20), (10, 20)], 0), ([(10, 20), (5, 10)], 0), ([(10, 20), (0, 30)], 100), ([(10, 20), (40, 30)], 0)]:
            with self.subTest(pair=pair), patch.object(tb, 'read_proc_stat_cpu_totals', side_effect=pair), patch.object(tb, 'time', SimpleNamespace(sleep=lambda delay: None)):
                self.assertEqual(tb.read_cpu_percent(), expected)
    def test_thermal_cpu_merged_with_hwmon(self):
        temps = ClosedFS([('nvme', [(1, 'Composite', '40000')])], [('x86_pkg_temp', '70000')]).collect()
        self.assertEqual([(t.source, t.label, t.celsius) for t in temps], [('hwmon', 'nvme:Composite', 40.0), ('thermal', 'x86_pkg_temp', 70.0)])
    def test_cooler_cpu_beats_hotter_non_cpu(self):
        cpu = self.T('thermal', 'cpu_thermal', 40)
        other = [self.T('hwmon', 'nvme:Composite', 90), self.T('hwmon', 'pch:temp1', 80)]
        for values in [other + [cpu], [cpu] + other]:
            with self.subTest(order=[t.label for t in values]):
                self.assertIs(tb.select_display_temperature(values), cpu)
    def test_non_cpu_only_is_unknown(self):
        for source, label in [('hwmon', 'nvme:Composite'), ('hwmon', 'pch:temp1'), ('thermal', 'acpitz'), ('thermal', 'soc_thermal'), ('hwmon', 'battery:CPU'), ('hwmon', 'dell_smm:temp1'), ('hwmon', 'coretemp:bogus')]:
            with self.subTest(label=label):
                self.assertIsNone(tb.select_display_temperature([self.T(source, label, 90)]))
    def test_known_hwmon_cpu_controls(self):
        for label in ['coretemp:Package id 0', 'coretemp:Core 2', 'coretemp:temp3', 'k10temp:Tctl', 'k10temp:Tdie', 'zenpower:Tctl', 'zenpower:Tdie', 'dell_smm:CPU']:
            cpu = self.T('hwmon', label, 42)
            with self.subTest(label=label):
                self.assertIs(tb.select_display_temperature([self.T('hwmon', 'nvme:Composite', 99), cpu]), cpu)
        for driver in ['k10temp', 'zenpower']:
            tctl, tdie = self.T('hwmon', driver + ':Tctl', 40), self.T('hwmon', driver + ':Tdie', 80)
            self.assertIs(tb.select_display_temperature([tdie, tctl]), tctl)
    def test_package_identity_and_source_are_anchored(self):
        for source, label in [('thermal', 'coretemp:Package id 0'), ('hwmon', 'notcoretemp:Core 0'), ('hwmon', 'coretemp:Package id x')]:
            with self.subTest(source=source, label=label):
                self.assertIsNone(tb.select_display_temperature([self.T(source, label, 80)]))
        cool, hot = self.T('hwmon', 'coretemp:Package id 1', 40), self.T('hwmon', 'coretemp:Package id 2', 70)
        self.assertIs(tb.select_display_temperature([cool, hot]), hot)
    def test_nonfinite_cpu_cannot_shadow_fallback(self):
        for value in [float('nan'), float('inf'), -float('inf')]:
            fallback = self.T('thermal', 'cpu-thermal', 40)
            with self.subTest(value=value):
                self.assertIs(tb.select_display_temperature([self.T('hwmon', 'coretemp:Package id 0', value), fallback]), fallback)
    def test_thermal_unit_is_millidegree(self):
        for raw, want in [('100', 0.1), ('0', 0.0), ('-1000', -1.0)]:
            with self.subTest(raw=raw):
                temps = ClosedFS(thermal=[('cpu_thermal', raw)]).collect()
                self.assertEqual([(t.source, t.label, t.celsius) for t in temps], [('thermal', 'cpu_thermal', want)])
    def test_same_file_different_identity_and_unreadable_alias(self):
        aliases = {'/sys/class/hwmon/hwmon0/temp1_input': '/synthetic/shared', '/sys/class/thermal/thermal_zone0/temp': '/synthetic/shared'}
        for raw, expected in [('40000', [('hwmon', 'board:temp1', 40.0), ('thermal', 'x86_pkg_temp', 40.0)]), (None, [('thermal', 'x86_pkg_temp', 40.0)])]:
            with self.subTest(raw=raw):
                temps = ClosedFS([('board', [(1, None, raw)])], [('x86_pkg_temp', '40000')], aliases).collect()
                self.assertEqual([(t.source, t.label, t.celsius) for t in temps], expected)
                self.assertEqual(tb.select_display_temperature(temps).label, 'x86_pkg_temp')
    def test_malformed_unreadable_controls(self):
        temps = ClosedFS([('nvme', [(1, None, None), (2, None, ''), (3, None, 'not-a-number')])], [('cpu_thermal', None), ('cpu-thermal', 'bad')]).collect()
        self.assertEqual(temps, [])
        self.assertIsNone(tb.select_display_temperature(temps))
    def test_zero_negative_and_extreme_finite_are_eligible(self):
        for value in [0.0, -1000.0, 1000.0]:
            cpu = self.T('hwmon', 'coretemp:Package id 0', value)
            with self.subTest(value=value):
                self.assertIs(tb.select_display_temperature([cpu]), cpu)
    def test_wire_fields_and_unknown_placeholders_preserved(self):
        temps = [self.T('hwmon', 'pch:temp1', 55), self.T('hwmon', 'nvme:Composite', 60)]
        with ExitStack() as stack:
            for name, value in [('read_cpu_percent', 25), ('read_memory_percent', 35), ('read_battery_state', (80, 1)), ('read_fan_rpm', 1200), ('read_host_temperatures', temps)]:
                stack.enter_context(patch.object(tb, name, return_value=value))
            self.assertEqual(tb.format_pc_monitor_payload(None), 'CPUNAC|GPU--|25|35|0|80|1')
            self.assertEqual(tb.format_temp_bars_payload(None, temps), 'CPU0C|NV60C|0|55|60|80|1')
            self.assertEqual(tb.format_fedora_temp_payload(None, temps), 'CPU=0|PCH=55|NVME=60|LOAD=25|RAM=35|BAT=80|AC=1|FAN=1200')
            args = tb.parse_args(['--payload-format', 'fedora-temp', '--line-ending', 'crlf'])
            self.assertEqual(tb.format_payload(None, args), 'CPU=0|PCH=55|NVME=60|LOAD=25|RAM=35|BAT=80|AC=1|FAN=1200\r\n')
            args = tb.parse_args(['--payload-format', 'display', '--line-ending', 'lf'])
            self.assertEqual(tb.format_payload(None, args), 'FEDORA TEMP N/A\n')
    def test_non_cpu_thermal_policy_preserved(self):
        fs = ClosedFS([('nvme', [(1, 'Composite', '40000')])], [('pch', '90000'), ('acpitz', '80000'), ('cpu_thermal', '50000')])
        self.assertEqual([(t.source, t.label, t.celsius) for t in fs.collect()], [('hwmon', 'nvme:Composite', 40.0), ('thermal', 'cpu_thermal', 50.0)])
        legacy = ClosedFS(thermal=[('acpitz', '80000')]).collect()
        self.assertEqual([(t.source, t.label, t.celsius) for t in legacy], [('thermal', 'acpitz', 80.0)])
        self.assertIsNone(tb.select_display_temperature(legacy))
