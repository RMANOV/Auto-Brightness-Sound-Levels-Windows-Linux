"""Date-aware local zones and calendar-owned activation windows, without host I/O."""
import datetime as dt
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ab_sun_timezone', SOURCE / 'sunrise_sunset_calculator.py')
ssc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ssc
spec.loader.exec_module(ssc)
DAY = dt.date(2026, 10, 9)
REAL_DATETIME = dt.datetime


class FrozenDateTime(REAL_DATETIME):
    @classmethod
    def now(cls, tz=None):
        value = cls(2026, 10, 9, 0, 30)
        return value if tz is None else value.replace(tzinfo=ZoneInfo('Europe/Sofia')).astimezone(tz)


class TimezoneTests(unittest.TestCase):
    def cached(self, **kwargs):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        home = Path(temporary.name)
        path = home / '.config/adaptive-controller/location.conf'
        path.parent.mkdir(parents=True)
        raw = json.dumps({'latitude': 42.6977, 'longitude': 23.3219,
                          'timezone_offset': 2, 'city': 'Sofia', 'auto_detected': False})
        path.write_text(raw)
        with patch.object(ssc.Path, 'home', return_value=home), patch.object(
                ssc, '_local_timezone', return_value=ZoneInfo('Europe/Sofia'), create=True), patch.object(
                ssc.datetime, 'datetime', FrozenDateTime):
            calc = ssc.SunCalculator(**kwargs)
        self.assertEqual(path.read_text(), raw, 'Reading legacy cache must not rewrite it')
        return calc

    def test_legacy_sofia_cache_uses_requested_date_dst(self):
        calc = self.cached()
        self.assertEqual((calc.latitude, calc.longitude), (42.6977, 23.3219))
        for date, offset in [(DAY, 3), (dt.date(2026, 10, 24), 3),
                             (dt.date(2026, 10, 25), 2), (dt.date(2027, 1, 9), 2)]:
            with self.subTest(date=date):
                expected = ssc.SunCalculator(42.6977, 23.3219, offset).calculate_sun_times(date)
                self.assertEqual(calc.calculate_sun_times(date), expected)

    def test_explicit_numeric_offsets_survive_coordinate_cache_loading(self):
        for offset in (0, 2):
            with self.subTest(offset=offset):
                calc = self.cached(timezone_offset=offset)
                self.assertEqual(calc.timezone_offset, offset)
                expected = ssc.SunCalculator(42.6977, 23.3219, offset).calculate_sun_times(DAY)
                self.assertEqual(calc.calculate_sun_times(DAY), expected)

    def test_supplied_coordinates_are_not_replaced_by_cache(self):
        calc = self.cached(latitude=0, longitude=0)
        self.assertEqual((calc.latitude, calc.longitude), (0, 0))
        self.assertEqual(calc.calculate_sun_times(DAY),
                         ssc.SunCalculator(0, 0, 3).calculate_sun_times(DAY))

    def test_complete_explicit_offset_never_resolves_host_timezone(self):
        with patch.object(ssc, '_local_timezone', side_effect=AssertionError('Host timezone read'), create=True):
            for offset in (0, 2):
                calc = ssc.SunCalculator(42.6977, 23.3219, offset)
                self.assertEqual(calc.timezone_offset, offset)
                self.assertIsInstance(calc.calculate_sun_times(DAY)['sunrise'], dt.time)

    def test_new_local_cache_keeps_coordinates_and_persists_zone_rules(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(
                ssc.Path, 'home', return_value=Path(temp)), patch.object(
                ssc, '_local_timezone', return_value=ZoneInfo('Europe/Sofia')):
            with patch.object(ssc.datetime, 'datetime', FrozenDateTime):
                calc = ssc.SunCalculator(42.6977, 23.3219)
            stored = json.loads((Path(temp) / '.config/adaptive-controller/location.conf').read_text())
            self.assertEqual(stored['timezone_name'], 'Europe/Sofia')
            self.assertEqual((stored['latitude'], stored['longitude']), (42.6977, 23.3219))
            winter = dt.date(2027, 1, 9)
            self.assertEqual(calc.calculate_sun_times(winter),
                             ssc.SunCalculator(42.6977, 23.3219, 2).calculate_sun_times(winter))
            with patch.object(ssc, '_local_timezone', side_effect=AssertionError('Saved zone lost')):
                reloaded = ssc.SunCalculator()
            self.assertEqual(reloaded.calculate_sun_times(winter), calc.calculate_sun_times(winter))

    def test_missing_local_zone_retains_cached_numeric_fallback(self):
        with patch.object(ssc, '_local_timezone', return_value=None, create=True):
            # cached() owns its resolver mock; test the fallback on its own temp cache.
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / '.config/adaptive-controller/location.conf'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({'latitude': 42.7, 'longitude': 23.3, 'timezone_offset': 2}))
                with patch.object(ssc.Path, 'home', return_value=Path(temp)):
                    calc = ssc.SunCalculator()
                self.assertEqual(calc.calculate_sun_times(DAY),
                                 ssc.SunCalculator(42.7, 23.3, 2).calculate_sun_times(DAY))

    def test_local_zone_discovery_reads_iana_link_without_commands(self):
        with patch.dict(ssc.os.environ, {}, clear=True), patch.object(
                ssc.Path, 'resolve', return_value=Path('/usr/share/zoneinfo/Europe/Sofia')):
            zone = ssc._local_timezone()
        self.assertEqual(REAL_DATETIME(2026, 10, 9, 12, tzinfo=zone).utcoffset(), dt.timedelta(hours=3))
        self.assertEqual(REAL_DATETIME(2026, 10, 25, 12, tzinfo=zone).utcoffset(), dt.timedelta(hours=2))

    def calendar_case(self, previous, today, following):
        calc = ssc.SunCalculator(42.7, 23.3, 2)
        values = {DAY - dt.timedelta(days=1): previous, DAY: today,
                  DAY + dt.timedelta(days=1): following}
        def solar(date=None):
            sunrise, sunset = values[date or DAY]
            return {'sunrise': sunrise, 'sunset': sunset, 'solar_noon': dt.time(12)}
        calc.calculate_sun_times = solar
        return calc

    def test_after_midnight_uses_previous_days_actual_sunset(self):
        calc = self.calendar_case((dt.time(7), dt.time(23, 30)),
                                  (dt.time(7), dt.time(20)), (dt.time(7), dt.time(20)))
        with patch.object(ssc.datetime, 'datetime', FrozenDateTime):
            self.assertEqual(calc.is_in_active_window(dt.time(0, 30)), (True, 'sunset'))
            self.assertEqual(calc.is_in_active_window(), (True, 'sunset'))
            self.assertEqual(calc.is_in_active_window(dt.time(1, 30, 1)), (False, None))

    def test_todays_late_sunset_does_not_activate_previous_midnight(self):
        calc = self.calendar_case((dt.time(7), dt.time(20)),
                                  (dt.time(7), dt.time(23, 30)), (dt.time(7), dt.time(20)))
        with patch.object(ssc.datetime, 'datetime', FrozenDateTime):
            self.assertEqual(calc.is_in_active_window(dt.time(0, 30)), (False, None))

    def test_before_midnight_uses_next_days_actual_sunrise(self):
        calc = self.calendar_case((dt.time(7), dt.time(18)),
                                  (dt.time(7), dt.time(18)), (dt.time(0, 15), dt.time(18)))
        with patch.object(ssc.datetime, 'datetime', FrozenDateTime):
            self.assertEqual(calc.is_in_active_window(dt.time(23, 45)), (True, 'sunrise'))
            self.assertEqual(calc.is_in_active_window(dt.time(23, 44, 59)), (False, None))

    def test_explicit_date_forwards_each_adjacent_event_date_once(self):
        calc = ssc.SunCalculator(42.7, 23.3, 2)
        target = dt.date(2026, 10, 25)
        with patch.object(calc, 'calculate_sun_times', return_value={
                'sunrise': dt.time(7), 'sunset': dt.time(19), 'solar_noon': dt.time(12)}) as solar:
            self.assertEqual(calc.is_in_active_window(dt.time(7), date=target), (True, 'sunrise'))
        self.assertEqual([call.args[0] for call in solar.call_args_list],
                         [target - dt.timedelta(days=1), target, target + dt.timedelta(days=1)])

    def test_two_hour_window_does_not_gain_an_hour_when_dst_ends(self):
        calc = self.cached()
        target = dt.date(2026, 10, 25)
        with patch.object(calc, 'calculate_sun_times', return_value={
                'sunrise': dt.time(2, 30), 'sunset': dt.time(19), 'solar_noon': dt.time(12)}):
            # 02:30 EEST + two elapsed hours is the second 03:30 (EET).
            windows = calc.get_activation_windows(target)
            self.assertEqual(windows['sunrise_end'], dt.time(3, 30))
            self.assertEqual(windows['sunrise_end'].fold, 1)
            self.assertEqual(calc.is_in_active_window(dt.time(3, 30, fold=1), target),
                             (True, 'sunrise'))
            self.assertEqual(calc.is_in_active_window(dt.time(3, 30, 1, fold=1), target),
                             (False, None))
