#!/bin/bash
# One command to bring a local Linux install up to date: pull main, rebuild the
# Rust controller and restart it with the new code. The 10-minute cron check
# already calls the manager, so the new schedule needs no crontab change.
# Usage: scripts/update-local.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANAGER="$REPO/adaptive_controller_manager.sh"

step() { printf '\n==> %s\n' "$*"; }

step "Updating $REPO from origin/main"
git -C "$REPO" fetch origin main
if [[ "$(git -C "$REPO" rev-parse --abbrev-ref HEAD)" != "main" ]]; then
    git -C "$REPO" checkout main
fi
git -C "$REPO" merge --ff-only origin/main

step "Building the release controller"
(cd "$REPO/adaptive-rust" && cargo build --release -p adaptive-controller --locked)

step "Restarting the controller with the new code"
restart_rc=0
"$MANAGER" restart || restart_rc=$?
case "$restart_rc" in
    0) echo "Restarted: the new controller adjusted brightness now." ;;
    1) echo "Restarted: no adjustment needed right now; the next 10-minute check runs the new code." ;;
    *) echo "Restart failed (exit $restart_rc); see ${XDG_STATE_HOME:-$HOME/.local/state}/adaptive-controller/controller.log" >&2 ;;
esac

# install_crontab.sh replaces the whole crontab, so it is never run from here.
step "Checking the 10-minute cron entry"
if crontab -l 2>/dev/null | grep -Fq "\"$MANAGER\" check"; then
    echo "OK: cron runs the manager every 10 minutes (sunrise/sunset windows and the periods between them)."
else
    echo "No 10-minute cron entry for $MANAGER. Run $REPO/install_crontab.sh once (it replaces your whole crontab)." >&2
fi

[[ "$restart_rc" -le 1 ]]
