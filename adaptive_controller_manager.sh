#!/bin/bash

# Adaptive Brightness & Volume Controller Manager
# Intelligent process management script for optimal performance and energy efficiency
# Optimized for Fedora 42 laptop systems

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/adaptive_brightness_volume.py"
RUST_BINARY="$SCRIPT_DIR/adaptive-rust/target/release/adaptive-controller"
USE_RUST=""
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/adaptive-controller"
RUN_TMP_DIR="$STATE_DIR/tmp"
LOCK_FILE="$STATE_DIR/controller.lock"
LOG_FILE="$STATE_DIR/controller.log"
PID_FILE="$STATE_DIR/controller.pid"
MAX_LOG_SIZE=1048576  # 1MB
AMBIENT_STATE="$HOME/.config/adaptive-controller/ambient_state.json"
PROBE_FILE=""
RUN_LOG=""
CONTROL_PID=""
OWNED_LOCK=false
CONTROLLER_TIMEOUT="${ADAPTIVE_CONTROLLER_TIMEOUT:-480s}"

# Check for --use-rust flag
for arg in "$@"; do
    if [[ "$arg" == "--use-rust" ]]; then
        USE_RUST=true
        # Remove the flag from arguments
        set -- "${@/--use-rust/}"
        break
    fi
done

# Auto-detect Rust binary if available
if [[ -x "$RUST_BINARY" ]]; then
    # Use Rust by default if available and not explicitly disabled
    if [[ "${PREFER_RUST:-true}" == "true" && "$USE_RUST" != "false" ]]; then
        USE_RUST=true
    fi
fi

# Intelligent time-based configuration
CURRENT_HOUR=$(date +%H)
CURRENT_DAY=$(date +%u)  # 1=Monday, 7=Sunday

# Function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
    
    # Rotate log if too large
    if [[ -f "$LOG_FILE" ]] && [[ $(stat -c%s "$LOG_FILE") -gt $MAX_LOG_SIZE ]]; then
        tail -n 100 "$LOG_FILE" > "${LOG_FILE}.tmp"
        mv "${LOG_FILE}.tmp" "$LOG_FILE"
        log_message "Log rotated due to size limit"
    fi
}

process_start_ticks() {
    local stat_line stat_tail
    local -a fields
    IFS= read -r stat_line < "/proc/$1/stat" 2>/dev/null || return 1
    stat_tail="${stat_line##*) }"
    read -r -a fields <<< "$stat_tail"
    printf '%s' "${fields[19]}"
}

# Verify exact supervisor argv, executable and process generation before any signal.
is_controller_running() {
    [[ -f "$PID_FILE" ]] || return 1
    local pid recorded_start actual_start executable
    local -a command_args
    read -r pid recorded_start < "$PID_FILE"
    if [[ "$pid" =~ ^[0-9]+$ && "$recorded_start" =~ ^[0-9]+$ ]] &&
       kill -0 "$pid" 2>/dev/null && [[ -r "/proc/$pid/cmdline" ]]; then
        actual_start=$(process_start_ticks "$pid")
        executable=$(readlink -f "/proc/$pid/exe" 2>/dev/null)
        mapfile -d '' -t command_args < "/proc/$pid/cmdline"
        if [[ "$recorded_start" != "0" && "$actual_start" == "$recorded_start" &&
              "$executable" == "$(readlink -f "$(command -v timeout)")" &&
              "${#command_args[@]}" -eq 4 &&
              "${command_args[1]}" == "--kill-after=2s" &&
              "${command_args[2]}" =~ ^[0-9]+([.][0-9]+)?[smhd]?$ &&
              "${command_args[3]}" == "$RUST_BINARY" ]]; then
            return 0
        fi
    fi
    rm -f "$PID_FILE"
    return 1
}

# Only a session/group whose leader is the recorded controller PID is targeted.
terminate_controller() {
    local pid="$1" group target
    group=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    target="$pid"
    [[ "$group" == "$pid" ]] && target="-$pid"
    kill -TERM -- "$target" 2>/dev/null || true
    for _ in {1..20}; do
        kill -0 -- "$target" 2>/dev/null || return 0
        sleep 0.1
    done
    kill -KILL -- "$target" 2>/dev/null || true
}

cleanup_owned_controller() {
    if [[ -n "$CONTROL_PID" ]]; then
        if [[ "$(ps -o ppid= -p "$CONTROL_PID" 2>/dev/null | tr -d ' ')" == "$$" ]]; then
            terminate_controller "$CONTROL_PID"
            # Never block cleanup indefinitely on a stuck native/kernel process.
            local child_state
            child_state=$(ps -o stat= -p "$CONTROL_PID" 2>/dev/null)
            if [[ -z "$child_state" || "$child_state" == Z* ]]; then
                wait "$CONTROL_PID" 2>/dev/null || true
            fi
        fi
        local recorded_pid recorded_start
        if [[ -f "$PID_FILE" ]]; then
            read -r recorded_pid recorded_start < "$PID_FILE"
            [[ "$recorded_pid" == "$CONTROL_PID" ]] && rm -f "$PID_FILE"
        fi
        CONTROL_PID=""
    fi
    if [[ "$OWNED_LOCK" == "true" && -f "$LOCK_FILE" && "$(cat "$LOCK_FILE")" == "$$" ]]; then
        rm -f "$LOCK_FILE"
    fi
    OWNED_LOCK=false
}

cleanup_manager() {
    cleanup_owned_controller
    if [[ -n "$RUN_LOG" && -f "$RUN_LOG" ]]; then
        cat "$RUN_LOG" >> "$LOG_FILE"
        rm -f "$RUN_LOG"
        RUN_LOG=""
    fi
    if [[ -n "$PROBE_FILE" ]]; then
        rm -f "$PROBE_FILE" "$PROBE_FILE.err"
    fi
}

# Cron does not inherit the desktop environment needed by PipeWire/PulseAudio.
restore_session_runtime() {
    [[ -z "${XDG_RUNTIME_DIR:-}" ]] || return 0
    local runtime_path
    runtime_path=$(timeout --kill-after=1s 2s loginctl show-user "$UID" -p RuntimePath --value 2>/dev/null) || return 0
    if [[ "$runtime_path" == /* && -d "$runtime_path" && -O "$runtime_path" ]]; then
        export XDG_RUNTIME_DIR="$runtime_path"
    fi
}

# A probe error is distinct from a valid unchanged ambient reading.
flash_detection_check() {
    [[ -n "$PROBE_FILE" ]] || PROBE_FILE=$(mktemp "$RUN_TMP_DIR/probe.XXXXXXXX") || return 2
    timeout --kill-after=2s 6s python3 -u "$SCRIPT_DIR/camera_probe.py" --state "$AMBIENT_STATE" > "$PROBE_FILE" 2> "$PROBE_FILE.err"
    local probe_rc=$?
    local result
    if [[ "$probe_rc" -gt 1 ]] || [[ ! -s "$PROBE_FILE" ]]; then
        log_message "Probe ERROR (exit $probe_rc): $(tail -c 1500 "$PROBE_FILE" "$PROBE_FILE.err" 2>/dev/null)"
        return 2
    fi
    result=$(command python3 "$SCRIPT_DIR/camera_probe.py" --state "$AMBIENT_STATE" --validate "$PROBE_FILE")
    if [[ $? -ne 0 ]] || [[ "$result" != "CHANGED" && "$result" != "UNCHANGED" ]] ||
       [[ "$result" == "CHANGED" && "$probe_rc" -ne 0 ]] ||
       [[ "$result" == "UNCHANGED" && "$probe_rc" -ne 1 ]]; then
        log_message "Probe ERROR: invalid measurement or inconsistent status: $result"
        return 2
    fi
    log_message "Ambient probe: $result"
    [[ "$result" == "CHANGED" ]] && return 0
    return 1
}

# Function to determine if we should be active based on sunrise/sunset times
should_be_active() {
    # First check if we're in a sunrise/sunset activation window
    local sunrise_sunset_status
    local window_type
    
    # Call the sunrise/sunset calculator
    local calculator_output
    calculator_output=$("$SCRIPT_DIR/sunrise_sunset_calculator.py" --check-active 2>/dev/null)
    
    if [[ $? -eq 0 && -n "$calculator_output" ]]; then
        if [[ "$calculator_output" == "ACTIVE:"* ]]; then
            window_type=$(echo "$calculator_output" | cut -d':' -f2)
            log_message "Sunrise/sunset window: Currently in $window_type activation window"
        else
            log_message "Sunrise/sunset window: Outside activation windows - skipping"
            return 1  # Outside sunrise/sunset activation windows
        fi
    else
        log_message "Schedule ERROR: sunrise/sunset calculation unavailable"
        return 2
    fi

    # Check if user session is active (for KDE Plasma)
    if ! loginctl show-session $(loginctl list-sessions | grep $(whoami) | awk '{print $1}') -p Active 2>/dev/null | grep -q "Active=yes"; then
        log_message "User session not active - skipping"
        return 1  # No active user session
    fi
    
    # Check if display is on (prevent running on locked screen without display)
    if command -v xset >/dev/null 2>&1; then
        if xset q | grep "Monitor is Off" >/dev/null 2>&1; then
            log_message "Display is off - skipping"
            return 1  # Display is off
        fi
    fi
    
    flash_detection_check
    return $?

}

# Function to check system load and resources
system_load_acceptable() {
    # Check if system load is reasonable (load average < 80% of CPU cores)
    local cpu_cores=$(nproc)
    local load_threshold=$(echo "$cpu_cores * 0.8" | bc -l)
    local current_load=$(cat /proc/loadavg | awk '{print $1}')
    
    if (( $(echo "$current_load > $load_threshold" | bc -l) )); then
        log_message "System load too high: $current_load > $load_threshold"
        return 1
    fi
    
    # Check available memory (require at least 200MB free)
    local mem_available=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    if [[ $mem_available -lt 204800 ]]; then  # 200MB in KB
        log_message "Insufficient memory: ${mem_available}KB available"
        return 1
    fi
    
    return 0
}

# The manager owns and waits for its child; a timeout is not convergence.
start_controller() {
    local activation_already_checked="${1:-false}"
    if is_controller_running; then
        log_message "Controller already running"
        return 0
    fi
    if [[ "$activation_already_checked" != "true" ]]; then
        should_be_active
        local activation_rc=$?
        [[ "$activation_rc" -eq 0 ]] || return "$activation_rc"
    fi
    if ! system_load_acceptable; then
        log_message "System resources unavailable; adjustment deferred"
        return 1
    fi
    if [[ -f "$LOCK_FILE" ]]; then
        local lock_pid
        lock_pid=$(cat "$LOCK_FILE")
        if [[ "$lock_pid" =~ ^[0-9]+$ ]] && ! kill -0 "$lock_pid" 2>/dev/null; then
            rm -f "$LOCK_FILE"
        fi
    fi
    if ! (set -C; echo $$ > "$LOCK_FILE") 2>/dev/null; then
        log_message "Another instance is starting (lock file exists)"
        return 1
    fi
    OWNED_LOCK=true
    if [[ "$USE_RUST" != "true" || ! -x "$RUST_BINARY" ]]; then
        log_message "Controller ERROR: verified Linux Rust backend is unavailable"
        rm -f "$LOCK_FILE"
        return 2
    fi
    local run_log
    run_log=$(mktemp "$RUN_TMP_DIR/run.XXXXXXXX") || { rm -f "$LOCK_FILE"; return 2; }
    RUN_LOG="$run_log"
    setsid timeout --kill-after=2s "$CONTROLLER_TIMEOUT" "$RUST_BINARY" > "$run_log" 2>&1 &
    local controller_pid=$!
    CONTROL_PID=$controller_pid
    printf '%s %s\n' "$controller_pid" "$(process_start_ticks "$controller_pid" || echo 0)" > "$PID_FILE"
    log_message "Controller started (PID: $controller_pid); awaiting actual completion"
    wait "$controller_pid"
    local exit_code=$?
    cat "$run_log" >> "$LOG_FILE"
    local converged=false
    if [[ "$exit_code" -eq 0 ]] && grep -q 'Converged in ' "$run_log"; then
        converged=true
    fi
    rm -f "$run_log"
    RUN_LOG=""
    cleanup_owned_controller
    if [[ "$converged" != "true" ]]; then
        log_message "Controller ERROR: no confirmed convergence (exit $exit_code)"
        [[ "$exit_code" -eq 0 ]] && return 2
        return "$exit_code"
    fi
    if [[ -z "$PROBE_FILE" ]] || ! command python3 "$SCRIPT_DIR/camera_probe.py" --state "$AMBIENT_STATE" --accept "$PROBE_FILE" >> "$LOG_FILE" 2>&1; then
        log_message "Controller ERROR: completed adjustment could not acknowledge ambient reference"
        return 2
    fi
    log_message "Controller completed with verified brightness; ambient reference accepted"
    return 0
}

# Stop only the validated controller and its owned process group.
stop_controller() {
    if ! is_controller_running; then
        log_message "Controller not running"
        return 0
    fi
    local pid
    read -r pid _ < "$PID_FILE"
    log_message "Stopping controller (PID: $pid)"
    terminate_controller "$pid"
    # The owning manager normally removes its PID on wait completion.
    return 0
}

# Function to check controller health
health_check() {
    if ! is_controller_running; then
        return 1
    fi
    
    # Check if log file is being updated (activity indicator)
    if [[ -f "$LOG_FILE" ]]; then
        local last_modified=$(stat -c %Y "$LOG_FILE" 2>/dev/null || echo 0)
        local current_time=$(date +%s)
        local time_diff=$((current_time - last_modified))
        
        # If no log activity for more than 10 minutes, consider unhealthy
        if [[ $time_diff -gt 600 ]]; then
            log_message "Controller appears inactive (no log activity for ${time_diff}s)"
            return 1
        fi
    fi
    
    return 0
}

# Main logic
umask 077
mkdir -p "$STATE_DIR" "$RUN_TMP_DIR" || exit 2
restore_session_runtime
trap cleanup_manager EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
case "${1:-check}" in
    restart)
        stop_controller || exit 2
        exec 9>"${LOCK_FILE}.guard" || exit 2
        flock -w 3 9 || exit 2
        ;;
    check|start)
        exec 9>"${LOCK_FILE}.guard" || exit 2
        flock -n 9 || exit 0
        ;;
esac
case "${1:-check}" in
    "start") start_controller; result=$? ;;
    "stop") stop_controller; result=$? ;;
    "restart") start_controller; result=$? ;;
    "status")
        if is_controller_running; then
            echo "Controller is running (PID: $(cat "$PID_FILE"))"
            result=0
        else
            echo "Controller is not running"
            result=1
        fi
        ;;
    "check")
        if is_controller_running; then
            log_message "Controller still running; skipping overlapping measurement"
            result=0
        else
            should_be_active
            result=$?
            if [[ "$result" -eq 0 ]]; then
                start_controller true
                result=$?
            elif [[ "$result" -eq 1 ]]; then
                result=0 # A policy skip is successful; ERROR remains nonzero.
            fi
        fi
        ;;
    *) echo "Usage: $0 {check|start|stop|restart|status}" >&2; result=2 ;;
esac
exit "$result"
