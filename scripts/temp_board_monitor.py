#!/usr/bin/env python3
"""Linux temperature sender for a small USB serial display board.

Stdlib-only by default. It reads host temperatures from sysfs and sends a
line-oriented status payload to a USB serial display board without pyserial.
"""

from __future__ import annotations

import argparse
import glob
import grp
import os
import pwd
import re
import select
import stat
import sys
import termios
import time
import tty
from dataclasses import dataclass
from pathlib import Path


CH340_VENDOR = "1a86"
CH340_PRODUCT = "7523"
DEFAULT_BAUD = 115200
TEMP_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


@dataclass
class Temperature:
    source: str
    label: str
    celsius: float


@dataclass
class SerialCandidate:
    path: str
    real_path: str
    vendor: str | None
    product: str | None
    by_id: bool
    temp_board_alias: bool
    readable: bool
    writable: bool
    owner: str
    group: str
    mode: str

    @property
    def is_ch340(self) -> bool:
        return (self.vendor or "").lower() == CH340_VENDOR and (
            self.product or ""
        ).lower() == CH340_PRODUCT

    @property
    def can_open(self) -> bool:
        return self.readable and self.writable


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def read_host_temperatures() -> list[Temperature]:
    temps: list[Temperature] = []
    seen_paths: set[Path] = set()

    for hwmon in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        chip_name = read_text(hwmon / "name") or hwmon.name
        for input_path in sorted(hwmon.glob("temp*_input")):
            seen_paths.add(input_path.resolve())
            raw = read_text(input_path)
            if raw is None:
                continue
            try:
                celsius = float(raw) / 1000.0
            except ValueError:
                continue
            prefix = input_path.name.removesuffix("_input")
            label = read_text(hwmon / f"{prefix}_label") or prefix
            temps.append(Temperature("hwmon", f"{chip_name}:{label}", celsius))

    if temps:
        return temps

    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        input_path = zone / "temp"
        if input_path.resolve() in seen_paths:
            continue
        raw = read_text(input_path)
        if raw is None:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        celsius = value / 1000.0 if abs(value) > 200 else value
        label = read_text(zone / "type") or zone.name
        temps.append(Temperature("thermal", label, celsius))

    return temps


def tty_sysfs_dir(dev_path: str) -> Path | None:
    tty_name = Path(os.path.realpath(dev_path)).name
    base = Path("/sys/class/tty") / tty_name
    return base if base.exists() else None


def usb_id_for_tty(dev_path: str) -> tuple[str | None, str | None]:
    base = tty_sysfs_dir(dev_path)
    if base is None:
        return None, None

    current = base.resolve()
    for candidate in [current, *current.parents]:
        vendor = read_text(candidate / "idVendor")
        product = read_text(candidate / "idProduct")
        if vendor or product:
            return vendor, product
    return None, None


def owner_group_mode(path: str) -> tuple[str, str, str]:
    try:
        st = os.stat(path)
    except OSError:
        return "?", "?", "?"
    try:
        owner = pwd.getpwuid(st.st_uid).pw_name
    except KeyError:
        owner = str(st.st_uid)
    try:
        group = grp.getgrgid(st.st_gid).gr_name
    except KeyError:
        group = str(st.st_gid)
    mode = stat.filemode(st.st_mode)
    return owner, group, mode


def discover_serial_candidates() -> list[SerialCandidate]:
    paths: list[tuple[str, bool, bool]] = []
    if Path("/dev/temp-board").exists():
        paths.append(("/dev/temp-board", False, True))

    for path in sorted(glob.glob("/dev/serial/by-id/*")):
        paths.append((path, True, False))

    for pattern in ("/dev/ttyUSB*", "/dev/ttyACM*"):
        for path in sorted(glob.glob(pattern)):
            paths.append((path, False, False))

    candidates: list[SerialCandidate] = []
    seen_real: set[str] = set()
    for path, by_id, alias in paths:
        real_path = os.path.realpath(path)
        key = real_path
        if key in seen_real and not alias and not by_id:
            continue
        seen_real.add(key)
        vendor, product = usb_id_for_tty(real_path)
        owner, group, mode = owner_group_mode(real_path)
        candidates.append(
            SerialCandidate(
                path=path,
                real_path=real_path,
                vendor=vendor,
                product=product,
                by_id=by_id,
                temp_board_alias=alias,
                readable=os.access(real_path, os.R_OK),
                writable=os.access(real_path, os.W_OK),
                owner=owner,
                group=group,
                mode=mode,
            )
        )

    candidates.sort(
        key=lambda item: (
            not item.temp_board_alias,
            not item.is_ch340,
            not item.by_id,
            not item.can_open,
            item.path,
        )
    )
    return candidates


def permission_guidance(candidate: SerialCandidate) -> str:
    return (
        f"{candidate.path} -> {candidate.real_path} is {candidate.mode} "
        f"{candidate.owner}:{candidate.group}; current user cannot open it. "
        "Fix options: add the user to dialout and relogin, install a udev rule "
        "for CH340, then unplug/replug the board. This script will not run sudo."
    )


def baud_constant(baud: int) -> int:
    name = f"B{baud}"
    value = getattr(termios, name, None)
    if value is None:
        raise ValueError(f"unsupported baud rate for termios: {baud}")
    return int(value)


def configure_serial(fd: int, baud: int) -> None:
    speed = baud_constant(baud)
    attrs = termios.tcgetattr(fd)
    tty.setraw(fd)
    attrs = termios.tcgetattr(fd)
    attrs[4] = speed
    attrs[5] = speed
    attrs[2] |= termios.CLOCAL | termios.CREAD
    attrs[2] &= ~termios.CSTOPB
    attrs[2] &= ~termios.PARENB
    attrs[2] &= ~termios.CSIZE
    attrs[2] &= ~termios.HUPCL
    attrs[2] |= termios.CS8
    attrs[6][termios.VMIN] = 0
    attrs[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW, attrs)


def open_serial(path: str, baud: int) -> int:
    fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    configure_serial(fd, baud)
    return fd


def write_serial_fd(fd: int, payload: str) -> int:
    data = payload.encode("utf-8", errors="replace")
    written = os.write(fd, data)
    termios.tcdrain(fd)
    return written


def write_serial_once(path: str, baud: int, payload: str, open_delay: float) -> int:
    fd = open_serial(path, baud)
    try:
        if open_delay > 0:
            time.sleep(open_delay)
        return write_serial_fd(fd, payload)
    finally:
        os.close(fd)


def read_serial_once(path: str, baud: int, timeout: float = 2.0) -> str | None:
    fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        configure_serial(fd, baud)
        deadline = time.monotonic() + timeout
        data = bytearray()
        while time.monotonic() < deadline:
            readable, _, _ = select.select([fd], [], [], 0.2)
            if not readable:
                continue
            chunk = os.read(fd, 256)
            if not chunk:
                continue
            data.extend(chunk)
            if b"\n" in data or b"\r" in data:
                break
        text = data.decode("utf-8", errors="replace").strip()
        return text or None
    finally:
        os.close(fd)


def parse_board_temperature(line: str | None) -> float | None:
    if not line:
        return None
    match = TEMP_RE.search(line)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def select_display_temperature(temps: list[Temperature]) -> Temperature | None:
    if not temps:
        return None

    preferred_labels = (
        "coretemp:Package id 0",
        "coretemp:Tctl",
        "coretemp:Tdie",
        "k10temp:Tctl",
        "k10temp:Tdie",
        "zenpower:Tctl",
        "zenpower:Tdie",
        "dell_smm:temp1",
    )
    for label in preferred_labels:
        for temp in temps:
            if temp.label == label:
                return temp

    core_temps = [temp for temp in temps if temp.label.startswith("coretemp:")]
    if core_temps:
        return max(core_temps, key=lambda temp: temp.celsius)
    return max(temps, key=lambda temp: temp.celsius)


def read_proc_stat_cpu_totals() -> tuple[int, int] | None:
    raw = read_text(Path("/proc/stat"))
    if not raw:
        return None
    first = raw.splitlines()[0].split()
    if not first or first[0] != "cpu":
        return None
    try:
        values = [int(value) for value in first[1:]]
    except ValueError:
        return None
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    total = sum(values)
    return idle, total


def read_cpu_percent(sample_delay: float = 0.15) -> int:
    before = read_proc_stat_cpu_totals()
    if before is None:
        return 0
    time.sleep(sample_delay)
    after = read_proc_stat_cpu_totals()
    if after is None:
        return 0
    idle_delta = after[0] - before[0]
    total_delta = after[1] - before[1]
    if total_delta <= 0:
        return 0
    return max(0, min(100, round((1.0 - idle_delta / total_delta) * 100)))


def read_memory_percent() -> int:
    raw = read_text(Path("/proc/meminfo"))
    if not raw:
        return 0
    values: dict[str, int] = {}
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                values[parts[0].rstrip(":")] = int(parts[1])
            except ValueError:
                pass
    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", 0)
    if total <= 0:
        return 0
    return max(0, min(100, round((1.0 - available / total) * 100)))


def read_battery_state() -> tuple[int, int]:
    capacity = 100
    for path in sorted(Path("/sys/class/power_supply").glob("BAT*/capacity")):
        raw = read_text(path)
        if raw is None:
            continue
        try:
            capacity = max(0, min(100, int(float(raw))))
            break
        except ValueError:
            continue

    ac_online = 1
    for path in sorted(Path("/sys/class/power_supply").glob("A*C*/online")):
        raw = read_text(path)
        if raw is None:
            continue
        try:
            ac_online = 1 if int(raw) else 0
            break
        except ValueError:
            continue
    return capacity, ac_online


def read_fan_rpm() -> int:
    for hwmon in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        for input_path in sorted(hwmon.glob("fan*_input")):
            raw = read_text(input_path)
            if raw is None:
                continue
            try:
                rpm = int(float(raw))
            except ValueError:
                continue
            if rpm > 0:
                return rpm
    return 0


def format_pc_monitor_payload(temp: Temperature | None) -> str:
    temp_text = "NA" if temp is None else str(round(temp.celsius))
    cpu_head = f"CPU{temp_text}C"[:8]
    gpu_head = "GPU--"
    cpu_pct = read_cpu_percent()
    ram_pct = read_memory_percent()
    gpu_pct = 0
    battery_pct, ac_online = read_battery_state()
    return f"{cpu_head}|{gpu_head}|{cpu_pct}|{ram_pct}|{gpu_pct}|{battery_pct}|{ac_online}"


def select_temp_by_label_fragment(temps: list[Temperature], fragment: str) -> Temperature | None:
    fragment = fragment.lower()
    matches = [temp for temp in temps if fragment in temp.label.lower()]
    if not matches:
        return None
    return max(matches, key=lambda temp: temp.celsius)


def format_temp_bars_payload(temp: Temperature | None, temps: list[Temperature]) -> str:
    cpu_temp = temp
    pch_temp = select_temp_by_label_fragment(temps, "pch")
    nvme_temp = select_temp_by_label_fragment(temps, "nvme")
    cpu_value = 0 if cpu_temp is None else max(0, min(100, round(cpu_temp.celsius)))
    pch_value = 0 if pch_temp is None else max(0, min(100, round(pch_temp.celsius)))
    nvme_value = 0 if nvme_temp is None else max(0, min(100, round(nvme_temp.celsius)))
    battery_pct, ac_online = read_battery_state()
    top_left = f"CPU{cpu_value}C"[:8]
    top_right = f"NV{nvme_value}C"[:8] if nvme_temp is not None else "NVME--"
    # The board firmware hardcodes labels and percent signs on the large rows.
    # In temp-bars mode those numeric bars intentionally represent Celsius.
    return f"{top_left}|{top_right}|{cpu_value}|{pch_value}|{nvme_value}|{battery_pct}|{ac_online}"


def format_fedora_temp_payload(temp: Temperature | None, temps: list[Temperature]) -> str:
    cpu_temp = 0 if temp is None else max(0, min(120, round(temp.celsius)))
    pch_temp = select_temp_by_label_fragment(temps, "pch")
    nvme_temp = select_temp_by_label_fragment(temps, "nvme")
    pch_value = 0 if pch_temp is None else max(0, min(120, round(pch_temp.celsius)))
    nvme_value = 0 if nvme_temp is None else max(0, min(120, round(nvme_temp.celsius)))
    cpu_load = read_cpu_percent()
    ram_pct = read_memory_percent()
    battery_pct, ac_online = read_battery_state()
    fan_rpm = read_fan_rpm()
    return (
        f"CPU={cpu_temp}|PCH={pch_value}|NVME={nvme_value}|"
        f"LOAD={cpu_load}|RAM={ram_pct}|BAT={battery_pct}|AC={ac_online}|FAN={fan_rpm}"
    )


def format_payload(temp: Temperature | None, args: argparse.Namespace) -> str:
    if args.message:
        text = args.message
    elif args.payload_format == "fedora-temp":
        text = format_fedora_temp_payload(temp, read_host_temperatures())
    elif args.payload_format == "pc-monitor":
        text = format_pc_monitor_payload(temp)
    elif args.payload_format == "temp-bars":
        text = format_temp_bars_payload(temp, read_host_temperatures())
    elif temp is None:
        text = "FEDORA TEMP N/A"
    elif args.payload_format == "numeric":
        text = f"{temp.celsius:.1f}"
    elif args.payload_format == "keyvalue":
        text = f"temp_c={temp.celsius:.1f} source={temp.label}"
    elif args.payload_format == "json":
        safe_label = temp.label.replace("\\", "\\\\").replace('"', '\\"')
        text = f'{{"host":"fedora","temp_c":{temp.celsius:.1f},"source":"{safe_label}"}}'
    else:
        text = f"FEDORA {temp.celsius:.1f}C"

    text = text.rstrip("\r\n")
    if args.line_ending == "crlf":
        text += "\r\n"
    elif args.line_ending == "cr":
        text += "\r"
    else:
        text += "\n"
    return text


def print_detection(candidates: list[SerialCandidate]) -> None:
    if not candidates:
        print("serial: no /dev/temp-board, /dev/serial/by-id, ttyUSB, or ttyACM devices found")
        return

    print("serial candidates:")
    for candidate in candidates:
        tags = []
        if candidate.temp_board_alias:
            tags.append("temp-board")
        if candidate.by_id:
            tags.append("by-id")
        if candidate.is_ch340:
            tags.append("CH340 1a86:7523")
        access = "openable" if candidate.can_open else "permission-denied"
        tag_text = f" ({', '.join(tags)})" if tags else ""
        print(
            f"  {candidate.path} -> {candidate.real_path}{tag_text}: "
            f"{access}, mode={candidate.mode}, owner={candidate.owner}, "
            f"group={candidate.group}, usb={candidate.vendor or '?'}:{candidate.product or '?'}"
        )
        if not candidate.can_open:
            print(f"    {permission_guidance(candidate)}")


def print_host_temperatures(temps: list[Temperature]) -> None:
    if not temps:
        print("host temp: no hwmon or thermal temperature sensors readable")
        return
    print("host temperatures:")
    for temp in temps:
        print(f"  {temp.label} [{temp.source}]: {temp.celsius:.1f} C")


def choose_candidate(candidates: list[SerialCandidate], requested_port: str | None) -> SerialCandidate | None:
    if requested_port:
        for candidate in candidates:
            if candidate.path == requested_port or candidate.real_path == os.path.realpath(requested_port):
                return candidate
        vendor, product = usb_id_for_tty(requested_port)
        owner, group, mode = owner_group_mode(requested_port)
        return SerialCandidate(
            path=requested_port,
            real_path=os.path.realpath(requested_port),
            vendor=vendor,
            product=product,
            by_id=requested_port.startswith("/dev/serial/by-id/"),
            temp_board_alias=requested_port == "/dev/temp-board",
            readable=os.access(requested_port, os.R_OK),
            writable=os.access(requested_port, os.W_OK),
            owner=owner,
            group=group,
            mode=mode,
        )
    return candidates[0] if candidates else None


def monitor(args: argparse.Namespace) -> int:
    serial_fd: int | None = None
    serial_real_path: str | None = None
    while True:
        temps = read_host_temperatures()
        candidates = discover_serial_candidates()
        display_temp = select_display_temperature(temps)
        if not args.quiet:
            print_host_temperatures(temps)
            if display_temp is None:
                print("display temp: no temperature selected")
            else:
                print(f"display temp: {display_temp.label} = {display_temp.celsius:.1f} C")
            print_detection(candidates)

        candidate = choose_candidate(candidates, args.port)
        if args.detect_only:
            return 0

        if candidate is None:
            print("serial send: no serial board detected")
        elif not candidate.can_open:
            print(f"serial send: {permission_guidance(candidate)}")
        elif args.dry_run:
            payload = format_payload(display_temp, args)
            action = "read" if args.read_board else "write"
            print(
                f"serial send: dry-run would {action} {candidate.path} at "
                f"{args.baud} baud; payload={payload!r}"
            )
        elif args.read_board:
            try:
                line = read_serial_once(candidate.real_path, args.baud)
            except PermissionError:
                print(f"board temp: {permission_guidance(candidate)}")
            except OSError as exc:
                print(f"board temp: cannot read {candidate.path}: {exc}")
            except ValueError as exc:
                print(f"board temp: {exc}")
                return 2
            else:
                value = parse_board_temperature(line)
                if value is None:
                    print(f"board temp: no numeric reading yet from {candidate.path}")
                else:
                    print(f"board temp [{candidate.path}]: {value:.1f} C")
        else:
            payload = format_payload(display_temp, args)
            try:
                if args.once:
                    written = write_serial_once(
                        candidate.real_path,
                        args.baud,
                        payload,
                        args.open_delay,
                    )
                else:
                    if serial_fd is None or serial_real_path != candidate.real_path:
                        if serial_fd is not None:
                            os.close(serial_fd)
                        serial_fd = open_serial(candidate.real_path, args.baud)
                        serial_real_path = candidate.real_path
                        if args.open_delay > 0:
                            time.sleep(args.open_delay)
                    written = write_serial_fd(serial_fd, payload)
            except PermissionError:
                print(f"serial send: {permission_guidance(candidate)}")
            except OSError as exc:
                print(f"serial send: cannot write {candidate.path}: {exc}")
                if serial_fd is not None:
                    try:
                        os.close(serial_fd)
                    except OSError:
                        pass
                serial_fd = None
                serial_real_path = None
            except ValueError as exc:
                print(f"serial send: {exc}")
                return 2
            else:
                if not args.quiet:
                    print(f"serial send [{candidate.path}]: wrote {written} bytes {payload!r}")

        if args.once:
            if serial_fd is not None:
                os.close(serial_fd)
            return 0
        print("")
        time.sleep(args.interval)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="run one sample and exit")
    parser.add_argument("--dry-run", action="store_true", help="detect but do not open serial")
    parser.add_argument("--detect-only", action="store_true", help="only print host/serial detection")
    parser.add_argument("--interval", type=float, default=5.0, help="loop interval in seconds")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="serial baud rate")
    parser.add_argument("--port", help="serial port path to use")
    parser.add_argument("--read-board", action="store_true", help="read one line from the board instead of sending Fedora temp")
    parser.add_argument(
        "--payload-format",
        choices=("fedora-temp", "pc-monitor", "temp-bars", "display", "numeric", "keyvalue", "json"),
        default="fedora-temp",
        help="serial payload format when sending Fedora temperature",
    )
    parser.add_argument("--message", help="override payload text to send to the board")
    parser.add_argument("--quiet", action="store_true", help="suppress repeated success logs in loop mode")
    parser.add_argument(
        "--line-ending",
        choices=("lf", "crlf", "cr"),
        default="crlf",
        help="line ending to append to serial payload",
    )
    parser.add_argument(
        "--open-delay",
        type=float,
        default=2.0,
        help="seconds to wait after opening serial before first write",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.interval <= 0:
        print("--interval must be positive", file=sys.stderr)
        return 2
    return monitor(args)


if __name__ == "__main__":
    raise SystemExit(main())
