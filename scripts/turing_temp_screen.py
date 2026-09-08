#!/usr/bin/env python3
"""Drive Turing/Turzx/XuanFang-style USB smart screens from Fedora temps."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import serial
from smartscreen_driver.lcd_comm import ComPortDetectError, LcdComm
from smartscreen_driver.lcd_comm import Orientation
from smartscreen_driver.lcd_comm_rev_a import LcdCommRevA
from smartscreen_driver.lcd_comm_rev_b import LcdCommRevB
from smartscreen_driver.lcd_comm_rev_c import LcdCommRevC
from smartscreen_driver.lcd_comm_rev_d import LcdCommRevD

import temp_board_monitor


REVISIONS = {
    "A": LcdCommRevA,
    "B": LcdCommRevB,
    "C": LcdCommRevC,
    "D": LcdCommRevD,
}


def open_serial_without_hardware_flow_control(self: LcdComm) -> None:
    if self.com_port == "AUTO":
        self.com_port = self.auto_detect_com_port()
        if not self.com_port:
            raise ComPortDetectError("No COM port detected")

    self.lcd_serial = serial.Serial(
        self.com_port,
        115200,
        timeout=1,
        write_timeout=2,
        rtscts=False,
        dsrdtr=False,
    )


LcdComm.open_serial = open_serial_without_hardware_flow_control

COLORS = {
    "A": (26, 79, 176),
    "B": (16, 128, 76),
    "C": (104, 54, 166),
    "D": (191, 104, 28),
}


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def current_temp_text() -> tuple[str, str]:
    temps = temp_board_monitor.read_host_temperatures()
    selected = temp_board_monitor.select_display_temperature(temps)
    if selected is None:
        return "FEDORA", "TEMP N/A"
    return "FEDORA", f"{selected.celsius:.1f} C"


def make_image(width: int, height: int, revision: str, note: str = "") -> Image.Image:
    bg = COLORS.get(revision, (30, 30, 30))
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    title_font = load_font(max(28, min(width, height) // 8))
    temp_font = load_font(max(42, min(width, height) // 5))
    small_font = load_font(max(18, min(width, height) // 15))

    title, temp = current_temp_text()
    lines = [title, temp, f"SMART REV {revision}"]
    if note:
        lines.append(note)

    y = int(height * 0.16)
    for index, line in enumerate(lines):
        font = temp_font if index == 1 else title_font if index == 0 else small_font
        bbox = draw.textbbox((0, 0), line, font=font)
        x = (width - (bbox[2] - bbox[0])) // 2
        draw.text((x, y), line, fill=(255, 255, 255), font=font)
        y += (bbox[3] - bbox[1]) + int(height * 0.05)
    return img


def drive_revision(revision: str, port: str, brightness: int, note: str = "") -> None:
    cls = REVISIONS[revision]
    lcd = cls(com_port=port)
    try:
        lcd.initialize_comm()
        lcd.screen_on()
        lcd.set_brightness(brightness)
        lcd.set_orientation(Orientation.PORTRAIT)
        img = make_image(lcd.width(), lcd.height(), revision, note)
        lcd.paint(img, (0, 0))
    finally:
        lcd.close_serial()


def probe(args: argparse.Namespace) -> int:
    for revision in args.revisions:
        print(f"Trying smart-screen revision {revision} on {args.port}")
        try:
            drive_revision(revision, args.port, args.brightness, "PROBE")
        except Exception as exc:
            print(f"  {revision}: {type(exc).__name__}: {exc}")
        time.sleep(args.probe_pause)
    return 0


def loop(args: argparse.Namespace) -> int:
    while True:
        try:
            drive_revision(args.revision, args.port, args.brightness)
        except Exception as exc:
            print(f"{args.revision}: {type(exc).__name__}: {exc}")
        if args.once:
            return 0
        time.sleep(args.interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/temp-board")
    parser.add_argument("--revision", choices=sorted(REVISIONS), default="A")
    parser.add_argument("--revisions", nargs="+", choices=sorted(REVISIONS), default=["A", "B", "C", "D"])
    parser.add_argument("--brightness", type=int, default=70)
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--probe-pause", type=float, default=8.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.probe:
        return probe(args)
    return loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
