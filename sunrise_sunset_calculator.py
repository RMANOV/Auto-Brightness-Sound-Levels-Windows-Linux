#!/usr/bin/env python3
"""
Sunrise/Sunset Calculator for Adaptive Controller
Pure Python implementation based on NOAA algorithms
Provides ±1.5 hour time windows around sunrise and sunset for intelligent scheduling
"""

import math
import datetime
import os
import json
import sys
from pathlib import Path


class SunCalculator:
    """Calculate sunrise and sunset times using NOAA algorithms"""
    
    def __init__(self, latitude=None, longitude=None, timezone_offset=None):
        """
        Initialize sun calculator with location coordinates
        
        Args:
            latitude: Latitude in decimal degrees (positive = North)
            longitude: Longitude in decimal degrees (positive = East) 
            timezone_offset: Hours offset from UTC (e.g., +2 for CEST)
        """
        self.latitude = latitude
        self.longitude = longitude
        self.timezone_offset = timezone_offset
        
        # Try to auto-detect location if not provided
        if not all([latitude, longitude, timezone_offset]):
            self._auto_detect_location()
    
    def _auto_detect_location(self):
        """Auto-detect location from config file or system timezone"""
        config_path = Path.home() / ".config" / "adaptive-controller" / "location.conf"
        
        # Try to load from config file
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.latitude = config.get('latitude')
                    self.longitude = config.get('longitude') 
                    self.timezone_offset = config.get('timezone_offset')
                    return
            except (json.JSONDecodeError, KeyError):
                pass
        
        # Fallback: Try timezone-based estimation
        self._estimate_from_timezone()
        
        # Create config directory and save detected location
        config_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_config(config_path)
    
    def _estimate_from_timezone(self):
        """Estimate location from system timezone (rough approximation)"""
        try:
            # Get system timezone offset
            now = datetime.datetime.now()
            utc_now = datetime.datetime.utcnow()
            self.timezone_offset = (now - utc_now).total_seconds() / 3600
            
            # Rough geographic estimates based on common timezones
            # These are very approximate but better than nothing
            timezone_map = {
                1: (52.5, 13.4),    # CET: Berlin-ish
                2: (50.0, 20.0),    # EET: Eastern Europe
                3: (55.7, 37.6),    # MSK: Moscow-ish
                -5: (40.7, -74.0),  # EST: New York-ish
                -8: (37.7, -122.4), # PST: San Francisco-ish
                0: (51.5, -0.1),    # GMT: London-ish
            }
            
            offset_key = round(self.timezone_offset)
            if offset_key in timezone_map:
                self.latitude, self.longitude = timezone_map[offset_key]
            else:
                # Default to Central Europe if unknown
                self.latitude, self.longitude = (50.0, 10.0)
                
        except Exception:
            # Ultimate fallback: Central Europe
            self.latitude = 50.0
            self.longitude = 10.0
            self.timezone_offset = 1.0
    
    def _save_config(self, config_path):
        """Save detected location to config file"""
        try:
            config = {
                'latitude': self.latitude,
                'longitude': self.longitude,
                'timezone_offset': self.timezone_offset,
                'auto_detected': True,
                'detection_date': datetime.datetime.now().isoformat()
            }
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception:
            pass  # Fail silently if can't save config
    
    def _julian_day(self, date):
        """Convert date to Julian day number"""
        a = (14 - date.month) // 12
        y = date.year + 4800 - a
        m = date.month + 12 * a - 3
        return date.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    
    def _julian_century(self, julian_day):
        """Convert Julian day to Julian century"""
        return (julian_day - 2451545.0) / 36525.0
    
    def _radians(self, degrees):
        """Convert degrees to radians"""
        return degrees * math.pi / 180.0
    
    def _degrees(self, radians):
        """Convert radians to degrees"""
        return radians * 180.0 / math.pi
    
    def _solar_mean_anomaly(self, t):
        """Calculate solar mean anomaly"""
        return 357.52911 + t * (35999.05029 - 0.0001537 * t)
    
    def _sun_equation_of_center(self, t):
        """Calculate sun's equation of center"""
        m = self._radians(self._solar_mean_anomaly(t))
        return (math.sin(m) * (1.914602 - t * (0.004817 + 0.000014 * t)) +
                math.sin(2 * m) * (0.019993 - 0.000101 * t) +
                math.sin(3 * m) * 0.000289)
    
    def _sun_true_longitude(self, t):
        """Calculate sun's true longitude"""
        l0 = 280.46646 + t * (36000.76983 + 0.0003032 * t)
        c = self._sun_equation_of_center(t)
        return l0 + c
    
    def _sun_apparent_longitude(self, t):
        """Calculate sun's apparent longitude"""
        omega = 125.04 - 1934.136 * t
        return self._sun_true_longitude(t) - 0.00569 - 0.00478 * math.sin(self._radians(omega))
    
    def _mean_obliquity_of_ecliptic(self, t):
        """Calculate mean obliquity of the ecliptic"""
        return (23.0 + (26.0 + ((21.448 - t * (46.8150 + t * (0.00059 - t * 0.001813)))) / 60.0) / 60.0)
    
    def _obliquity_correction(self, t):
        """Calculate obliquity correction"""
        e0 = self._mean_obliquity_of_ecliptic(t)
        omega = 125.04 - 1934.136 * t
        return e0 + 0.00256 * math.cos(self._radians(omega))
    
    def _sun_declination(self, t):
        """Calculate solar declination"""
        e = self._radians(self._obliquity_correction(t))
        lambda_sun = self._radians(self._sun_apparent_longitude(t))
        return self._degrees(math.asin(math.sin(e) * math.sin(lambda_sun)))
    
    def _equation_of_time(self, t):
        """Calculate equation of time in minutes"""
        epsilon = self._radians(self._obliquity_correction(t))
        l0 = self._radians(self._sun_true_longitude(t))
        e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
        m = self._radians(self._solar_mean_anomaly(t))
        
        y = math.tan(epsilon / 2.0) ** 2
        
        sin2l0 = math.sin(2.0 * l0)
        sinm = math.sin(m)
        cos2l0 = math.cos(2.0 * l0)
        sin4l0 = math.sin(4.0 * l0)
        sin2m = math.sin(2.0 * m)
        
        etime = (y * sin2l0 - 2.0 * e * sinm + 4.0 * e * y * sinm * cos2l0 -
                 0.5 * y * y * sin4l0 - 1.25 * e * e * sin2m)
        
        return self._degrees(etime) * 4.0
    
    def _hour_angle_sunrise(self, latitude, solar_dec):
        """Calculate hour angle for sunrise/sunset"""
        lat_rad = self._radians(latitude)
        sdec_rad = self._radians(solar_dec)
        
        try:
            ha_arg = (math.cos(self._radians(90.833)) / (math.cos(lat_rad) * math.cos(sdec_rad)) -
                     math.tan(lat_rad) * math.tan(sdec_rad))
            
            if ha_arg < -1.0:
                return 180.0  # Polar day
            elif ha_arg > 1.0:
                return 0.0    # Polar night
            else:
                return self._degrees(math.acos(ha_arg))
        except (ValueError, ZeroDivisionError):
            return 90.0  # Default to reasonable value
    
    def calculate_sun_times(self, date=None):
        """
        Calculate sunrise, sunset and solar noon times
        
        Args:
            date: datetime.date object (defaults to today)
            
        Returns:
            dict with keys: sunrise, sunset, solar_noon (all datetime.time objects)
        """
        if date is None:
            date = datetime.date.today()
        
        # Convert to Julian day and century
        jd = self._julian_day(date)
        t = self._julian_century(jd)
        
        # Calculate solar declination and equation of time
        solar_dec = self._sun_declination(t)
        eq_time = self._equation_of_time(t)
        
        # Calculate hour angle for sunrise/sunset
        ha_sunrise = self._hour_angle_sunrise(self.latitude, solar_dec)
        
        # Calculate times in minutes from midnight UTC
        time_correction = eq_time + 4 * self.longitude
        solar_noon_utc = 720 - time_correction
        sunrise_utc = solar_noon_utc - 4 * ha_sunrise
        sunset_utc = solar_noon_utc + 4 * ha_sunrise
        
        # Convert to local time
        local_offset_minutes = self.timezone_offset * 60
        
        def minutes_to_time(minutes):
            """Convert minutes from midnight to time object"""
            minutes += local_offset_minutes
            
            # Handle day overflow
            while minutes < 0:
                minutes += 1440
            while minutes >= 1440:
                minutes -= 1440
                
            hours = int(minutes // 60)
            mins = int(minutes % 60)
            return datetime.time(hours, mins)
        
        return {
            'sunrise': minutes_to_time(sunrise_utc),
            'sunset': minutes_to_time(sunset_utc),
            'solar_noon': minutes_to_time(solar_noon_utc)
        }
    
    def get_activation_windows(self, date=None):
        """
        Get sunrise and sunset activation windows (±1.5 hours)
        
        Args:
            date: datetime.date object (defaults to today)
            
        Returns:
            dict with keys: sunrise_start, sunrise_end, sunset_start, sunset_end
            All values are datetime.time objects
        """
        sun_times = self.calculate_sun_times(date)
        
        def add_hours_to_time(time_obj, hours):
            """Add hours to a time object"""
            dt = datetime.datetime.combine(datetime.date.today(), time_obj)
            dt += datetime.timedelta(hours=hours)
            return dt.time()
        
        return {
            'sunrise_start': add_hours_to_time(sun_times['sunrise'], -1.5),
            'sunrise_end': add_hours_to_time(sun_times['sunrise'], 1.5),
            'sunset_start': add_hours_to_time(sun_times['sunset'], -1.5),
            'sunset_end': add_hours_to_time(sun_times['sunset'], 1.5)
        }
    
    def is_in_active_window(self, current_time=None):
        """
        Check if current time is within sunrise/sunset activation windows
        
        Args:
            current_time: datetime.time object (defaults to now)
            
        Returns:
            tuple: (is_active, window_type) where window_type is 'sunrise', 'sunset', or None
        """
        if current_time is None:
            current_time = datetime.datetime.now().time()
        
        windows = self.get_activation_windows()
        
        # Check sunrise window
        if windows['sunrise_start'] <= current_time <= windows['sunrise_end']:
            return True, 'sunrise'
        
        # Check sunset window (handle day overflow)
        sunset_start = windows['sunset_start']
        sunset_end = windows['sunset_end']
        
        if sunset_start <= sunset_end:
            # Normal case: both times on same day
            if sunset_start <= current_time <= sunset_end:
                return True, 'sunset'
        else:
            # Day overflow case: sunset window crosses midnight
            if current_time >= sunset_start or current_time <= sunset_end:
                return True, 'sunset'
        
        return False, None


def main():
    """Command line interface for sunrise/sunset calculations"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate sunrise/sunset activation windows')
    parser.add_argument('--latitude', type=float, help='Latitude in decimal degrees')
    parser.add_argument('--longitude', type=float, help='Longitude in decimal degrees')
    parser.add_argument('--timezone', type=float, help='Timezone offset from UTC')
    parser.add_argument('--check-active', action='store_true', help='Check if currently in active window')
    parser.add_argument('--get-windows', action='store_true', help='Get today\'s activation windows')
    parser.add_argument('--get-times', action='store_true', help='Get today\'s sunrise/sunset times')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        # Initialize calculator
        calc = SunCalculator(args.latitude, args.longitude, args.timezone)
        
        if args.verbose:
            print(f"Location: {calc.latitude:.2f}°N, {calc.longitude:.2f}°E")
            print(f"Timezone: UTC{calc.timezone_offset:+.1f}")
            print()
        
        if args.check_active:
            is_active, window_type = calc.is_in_active_window()
            if is_active:
                print(f"ACTIVE:{window_type}")
            else:
                print("INACTIVE")
        
        elif args.get_windows:
            windows = calc.get_activation_windows()
            print(f"SUNRISE_WINDOW:{windows['sunrise_start']}:{windows['sunrise_end']}")
            print(f"SUNSET_WINDOW:{windows['sunset_start']}:{windows['sunset_end']}")
        
        elif args.get_times:
            times = calc.calculate_sun_times()
            print(f"SUNRISE:{times['sunrise']}")
            print(f"SUNSET:{times['sunset']}")
            print(f"SOLAR_NOON:{times['solar_noon']}")
        
        else:
            # Default: show all information
            times = calc.calculate_sun_times()
            windows = calc.get_activation_windows()
            is_active, window_type = calc.is_in_active_window()
            
            print(f"Sun Times for {datetime.date.today()}:")
            print(f"  Sunrise: {times['sunrise']}")
            print(f"  Solar Noon: {times['solar_noon']}")
            print(f"  Sunset: {times['sunset']}")
            print()
            print(f"Activation Windows (±1.5h):")
            print(f"  Morning: {windows['sunrise_start']} - {windows['sunrise_end']}")
            print(f"  Evening: {windows['sunset_start']} - {windows['sunset_end']}")
            print()
            if is_active:
                print(f"Status: ACTIVE ({window_type} window)")
            else:
                print("Status: INACTIVE")
    
    except Exception as e:
        print(f"ERROR:{e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()