import datetime as dt
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ab_sun', SOURCE / 'sunrise_sunset_calculator.py')
ssc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ssc
spec.loader.exec_module(ssc)


class SunTests(unittest.TestCase):
    def test_zero_coordinates_are_supplied_values_red(self):
        for values in [(0, 23.3, 2), (42.7, 0, 2), (42.7, 23.3, 0), (0, 0, 0)]:
            with self.subTest(values=values), patch.object(ssc.SunCalculator, '_auto_detect_location') as detect:
                calc = ssc.SunCalculator(*values)
                self.assertEqual((calc.latitude, calc.longitude, calc.timezone_offset), values)
                detect.assert_not_called()

    def test_missing_coordinates_still_detect_control(self):
        for values in [(None, 23.3, 2), (42.7, None, 2), (42.7, 23.3, None)]:
            with self.subTest(values=values), patch.object(ssc.SunCalculator, '_auto_detect_location') as detect:
                ssc.SunCalculator(*values)
                detect.assert_called_once_with()
        with patch.object(ssc.SunCalculator, '_auto_detect_location') as detect:
            ssc.SunCalculator(42.7, 23.3, 2)
            detect.assert_not_called()

    def calculator(self, sunrise, sunset):
        calc = ssc.SunCalculator(42.7, 23.3, 2)
        calc.calculate_sun_times = lambda date=None: {'sunrise': sunrise, 'sunset': sunset, 'solar_noon': dt.time(12)}
        return calc

    def test_default_window_lengths_and_date_forwarding_control(self):
        calc = ssc.SunCalculator(42.7, 23.3, 2)
        date = dt.date(2026, 1, 2)
        with patch.object(calc, 'calculate_sun_times', return_value={
            'sunrise': dt.time(0, 15), 'sunset': dt.time(12, 30), 'solar_noon': dt.time(6, 30)
        }) as solar:
            self.assertEqual(calc.get_activation_windows(date), {
                'sunrise_start': dt.time(22, 45), 'sunrise_end': dt.time(6, 15),
                'sunset_start': dt.time(11), 'sunset_end': dt.time(18, 30),
            })
            solar.assert_called_once_with(date)

    def test_midnight_sunrise_inclusive_red(self):
        calc = self.calculator(dt.time(0, 15), dt.time(12, 30))
        for value in [dt.time(22, 45), dt.time(23, 59), dt.time(0), dt.time(6, 15)]:
            with self.subTest(value=value):
                self.assertEqual(calc.is_in_active_window(value), (True, 'sunrise'))

    def test_midnight_sunrise_outside_control(self):
        calc = self.calculator(dt.time(0, 15), dt.time(12, 30))
        for value in [dt.time(22, 44, 59), dt.time(6, 15, 1)]:
            self.assertEqual(calc.is_in_active_window(value), (False, None))

    def test_existing_sunset_and_day_bounds_control(self):
        calc = self.calculator(dt.time(7), dt.time(19))
        for value, expected in [(dt.time(5, 29, 59), (False, None)),
                                (dt.time(5, 30), (True, 'sunrise')),
                                (dt.time(13), (True, 'sunrise')),
                                (dt.time(13, 0, 1), (False, None)),
                                (dt.time(17, 30), (True, 'sunset')),
                                (dt.time(1), (True, 'sunset')),
                                (dt.time(1, 0, 1), (False, None))]:
            with self.subTest(value=value):
                self.assertEqual(calc.is_in_active_window(value), expected)
        calc = self.calculator(dt.time(8), dt.time(23, 30))
        for value, expected in [(dt.time(21, 59, 59), (False, None)),
                                (dt.time(22), (True, 'sunset')),
                                (dt.time(0), (True, 'sunset')),
                                (dt.time(5, 30), (True, 'sunset')),
                                (dt.time(5, 30, 1), (False, None))]:
            with self.subTest(value=value):
                self.assertEqual(calc.is_in_active_window(value), expected)

    def test_overlap_retains_sunrise_first_control(self):
        calc = self.calculator(dt.time(8, 30), dt.time(9, 30))
        self.assertEqual(calc.is_in_active_window(dt.time(10)), (True, 'sunrise'))
