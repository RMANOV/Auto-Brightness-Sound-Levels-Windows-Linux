import sys
import time
from machine import I2C, Pin, UART
import ssd1306


i2c = I2C(scl=Pin(14), sda=Pin(12), freq=400000)
oled = ssd1306.SSD1306_I2C(128, 64, i2c)
uart = UART(0, baudrate=115200)


def clear_msg(line1, line2="", line3=""):
    oled.fill(0)
    oled.text(line1[:16], 0, 8)
    if line2:
        oled.text(line2[:16], 0, 24)
    if line3:
        oled.text(line3[:16], 0, 40)
    oled.show()


def clamp_int(value, default=0, lo=0, hi=120):
    try:
        value = int(float(value))
    except Exception:
        return default
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def parse_key_payload(line):
    data = {}
    for part in line.split("|"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        data[key.strip().upper()] = value.strip()
    if "CPU" not in data:
        return None
    return {
        "cpu": clamp_int(data.get("CPU")),
        "pch": clamp_int(data.get("PCH")),
        "nvme": clamp_int(data.get("NVME")),
        "load": clamp_int(data.get("LOAD"), hi=100),
        "ram": clamp_int(data.get("RAM"), hi=100),
        "bat": clamp_int(data.get("BAT"), default=100, hi=100),
        "ac": "1" if data.get("AC", "1") == "1" else "0",
        "fan": clamp_int(data.get("FAN"), hi=9999),
    }


def parse_legacy_payload(line):
    parts = line.split("|")
    if len(parts) != 7:
        return None
    return {
        "cpu": clamp_int(parts[2]),
        "pch": 0,
        "nvme": clamp_int(parts[4]),
        "load": clamp_int(parts[2], hi=100),
        "ram": clamp_int(parts[3], hi=100),
        "bat": clamp_int(parts[5], default=100, hi=100),
        "ac": "1" if parts[6] == "1" else "0",
        "fan": 0,
    }


def bar(x, y, w, pct):
    pct = clamp_int(pct, hi=100)
    oled.rect(x, y, w, 7, 1)
    fill = int((w - 2) * pct / 100)
    if fill > 0:
        oled.fill_rect(x + 1, y + 1, fill, 5, 1)


SEGMENTS = {
    "0": "ABCDEF",
    "1": "BC",
    "2": "ABGED",
    "3": "ABGCD",
    "4": "FGBC",
    "5": "AFGCD",
    "6": "AFGECD",
    "7": "ABC",
    "8": "ABCDEFG",
    "9": "ABFGCD",
}


def segment_digit(x, y, ch, w=18, h=34, t=4):
    segs = SEGMENTS.get(ch, "")
    if "A" in segs:
        oled.fill_rect(x + t, y, w - 2 * t, t, 1)
    if "B" in segs:
        oled.fill_rect(x + w - t, y + t, t, h // 2 - t, 1)
    if "C" in segs:
        oled.fill_rect(x + w - t, y + h // 2, t, h // 2 - t, 1)
    if "D" in segs:
        oled.fill_rect(x + t, y + h - t, w - 2 * t, t, 1)
    if "E" in segs:
        oled.fill_rect(x, y + h // 2, t, h // 2 - t, 1)
    if "F" in segs:
        oled.fill_rect(x, y + t, t, h // 2 - t, 1)
    if "G" in segs:
        oled.fill_rect(x + t, y + h // 2 - t // 2, w - 2 * t, t, 1)


def large_number_in_section(value, x0, width):
    text = str(clamp_int(value))
    if len(text) <= 2:
        digit_w = 14
        digit_h = 36
        thick = 3
        gap = 3
    else:
        digit_w = 10
        digit_h = 36
        thick = 2
        gap = 2
    total_w = len(text) * digit_w + (len(text) - 1) * gap
    x = x0 + max(0, (width - total_w) // 2)
    y = 20
    for ch in text:
        segment_digit(x, y, ch, digit_w, digit_h, thick)
        x += digit_w + gap


def section(x0, width, label, value):
    label_x = x0 + max(0, (width - len(label) * 8) // 2)
    oled.text(label, label_x, 2)
    large_number_in_section(value, x0, width)
    oled.text("C", x0 + width - 9, 52)


def draw_status(data):
    oled.fill(0)
    oled.vline(42, 0, 64, 1)
    oled.vline(85, 0, 64, 1)
    section(0, 42, "CPU", data["cpu"])
    section(43, 42, "PCH", data["pch"])
    section(86, 42, "NV", data["nvme"])
    oled.show()


clear_msg("THERMAL WATCH", "waiting serial", "115200 baud")
last_rx = time.ticks_ms()

while True:
    try:
        raw = sys.stdin.readline()
        if not raw:
            continue
        line = raw.strip()
        if not line:
            continue
        data = parse_key_payload(line) or parse_legacy_payload(line)
        if data is None:
            clear_msg("BAD PAYLOAD", line[:16], "need CPU=..")
            time.sleep(1)
            continue
        draw_status(data)
        last_rx = time.ticks_ms()
    except Exception as exc:
        clear_msg("ERROR", type(exc).__name__[:16], str(exc)[:16])
        time.sleep(2)
