#!/usr/bin/env bash
set -euo pipefail

RULE_FILE="/etc/udev/rules.d/99-temp-board-ch340.rules"
RULE='SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", GROUP="input", MODE="0660", TAG+="uaccess", SYMLINK+="temp-board"'

usage() {
  cat <<'EOF'
Usage: scripts/install_temp_board_udev.sh [--install]

Prints the CH340 temp-board udev rule by default.
Use --install to write it when running as root or when passwordless sudo is available.
This script does not prompt for a sudo password.
EOF
}

print_rule() {
  printf 'udev rule path: %s\n' "$RULE_FILE"
  printf '%s\n' "$RULE"
}

install_rule_as_root() {
  printf '%s\n' "$RULE" > "$RULE_FILE"
  udevadm control --reload-rules
  udevadm trigger --subsystem-match=tty
  printf 'Installed %s\n' "$RULE_FILE"
  printf 'Unplug/replug the CH340 board. Group=input makes it immediately usable for this Fedora user.\n'
}

install_rule() {
  print_rule
  if [[ "${EUID}" -eq 0 ]]; then
    install_rule_as_root
    return
  fi

  if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    printf '%s\n' "$RULE" | sudo tee "$RULE_FILE" >/dev/null
    sudo udevadm control --reload-rules
    sudo udevadm trigger --subsystem-match=tty
    printf 'Installed %s via sudo -n.\n' "$RULE_FILE"
    printf 'Unplug/replug the CH340 board. Group=input makes it immediately usable for this Fedora user.\n'
    return
  fi

  cat <<EOF
Not installed: root or passwordless sudo is required, and sudo -n is not available.

Run manually when ready:
  echo '$RULE' | sudo tee '$RULE_FILE'
  sudo udevadm control --reload-rules
  sudo udevadm trigger --subsystem-match=tty

Then unplug/replug the CH340 board.
EOF
  exit 2
}

case "${1:-}" in
  "")
    print_rule
    ;;
  --install)
    install_rule
    ;;
  -h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
