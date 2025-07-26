#!/bin/bash

# Adaptive Brightness & Volume Controller Manager
# Intelligent process management script for optimal performance and energy efficiency
# Optimized for Fedora 42 laptop systems

# Configuration
SCRIPT_DIR="/home/rmanov/Auto-Brightness-Sound-Levels-Windows-Linux"
PYTHON_SCRIPT="$SCRIPT_DIR/adaptive_brightness_volume.py"
LOCK_FILE="/tmp/adaptive_controller.lock"
LOG_FILE="/tmp/adaptive_controller.log"
PID_FILE="/tmp/adaptive_controller.pid"
MAX_LOG_SIZE=1048576  # 1MB

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

# Function to check if process is running and healthy
is_controller_running() {
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE")
        
        # Check if PID exists and is our process
        if kill -0 "$pid" 2>/dev/null; then
            # Verify it's actually our python script
            if pgrep -f "adaptive_brightness_volume.py" >/dev/null; then
                return 0  # Running
            fi
        fi
        
        # Stale PID file
        rm -f "$PID_FILE"
    fi
    return 1  # Not running
}

# Function to perform quick flash detection for significant changes
flash_detection_check() {
    # Quick 2-second environmental check for dramatic changes
    local temp_log="/tmp/flash_detection.log"
    
    # Get current saved state
    local saved_brightness=30
    local saved_volume=20
    
    if [[ -f "/home/rmanov/.config/adaptive-controller/last_state.txt" ]]; then
        saved_brightness=$(grep "brightness=" "/home/rmanov/.config/adaptive-controller/last_state.txt" | cut -d'=' -f2 2>/dev/null || echo 30)
        saved_volume=$(grep "volume=" "/home/rmanov/.config/adaptive-controller/last_state.txt" | cut -d'=' -f2 2>/dev/null || echo 20)
    fi
    
    # Quick environmental sampling (2 seconds only)
    cd "$SCRIPT_DIR"
    timeout 2s python3 -c "
import sys
sys.path.append('$SCRIPT_DIR')
try:
    from adaptive_brightness_volume import AdaptiveBrightnessVolumeController
    import cv2
    import numpy as np
    
    # Quick camera check
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_brightness = np.mean(gray) / 255 * 100
            
            # Calculate percentage change from saved state
            brightness_change = abs(current_brightness - $saved_brightness) / max($saved_brightness, 1) * 100
            
            print(f'flash_brightness_change:{brightness_change:.1f}')
        cap.release()
    cv2.destroyAllWindows()
except Exception as e:
    print(f'flash_error:{e}')
" > "$temp_log" 2>&1
    
    # Parse results
    if [[ -f "$temp_log" ]]; then
        local brightness_change=$(grep "flash_brightness_change:" "$temp_log" | cut -d':' -f2 2>/dev/null || echo 0)
        
        # Check if change is significant (>40%)
        if (( $(echo "$brightness_change > 40.0" | bc -l 2>/dev/null || echo 0) )); then
            log_message "Flash detection: Significant change detected (${brightness_change}% brightness change)"
            rm -f "$temp_log"
            return 0  # Significant change - should activate
        else
            log_message "Flash detection: No significant change (${brightness_change}% brightness change) - skipping activation"
            rm -f "$temp_log"
            return 1  # No significant change - skip activation
        fi
    fi
    
    rm -f "$temp_log"
    return 1  # Default to skip if detection failed
}

# Function to determine if we should be active based on time
should_be_active() {
    # Special deep night work mode (2 AM - 5 AM) - minimal disruption
    if [[ $CURRENT_HOUR -ge 2 && $CURRENT_HOUR -lt 5 ]]; then
        log_message "Deep night work mode (2-5 AM) - minimal activity"
        return 1  # Inactive during deep night work hours
    fi
    
    # Extended night hours check for sleep (11 PM - 7 AM)
    if [[ $CURRENT_HOUR -ge 23 || $CURRENT_HOUR -lt 7 ]]; then
        # Flash detection during extended night hours
        if ! flash_detection_check; then
            return 1  # No significant changes during night
        fi
        log_message "Night mode: Significant environmental change detected - activating"
    fi
    
    # Check if user session is active (for KDE Plasma)
    if ! loginctl show-session $(loginctl list-sessions | grep $(whoami) | awk '{print $1}') -p Active 2>/dev/null | grep -q "Active=yes"; then
        return 1  # No active user session
    fi
    
    # Check if display is on (prevent running on locked screen without display)
    if command -v xset >/dev/null 2>&1; then
        if xset q | grep "Monitor is Off" >/dev/null 2>&1; then
            return 1  # Display is off
        fi
    fi
    
    # During normal hours, perform flash detection before activation
    if ! flash_detection_check; then
        return 1  # No significant environmental changes detected
    fi
    
    return 0  # Should be active - significant changes detected
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

# Function to start the controller
start_controller() {
    if is_controller_running; then
        log_message "Controller already running"
        return 0
    fi
    
    # Check prerequisites
    if ! should_be_active; then
        log_message "Should not be active at this time ($(date '+%a %H:%M'))"
        return 1
    fi
    
    if ! system_load_acceptable; then
        log_message "System load too high, skipping start"
        return 1
    fi
    
    # Create lock file
    if ! (set -C; echo $$ > "$LOCK_FILE") 2>/dev/null; then
        log_message "Another instance is starting (lock file exists)"
        return 1
    fi
    
    log_message "Starting adaptive controller..."
    
    # Change to script directory
    cd "$SCRIPT_DIR" || {
        log_message "Failed to change to script directory: $SCRIPT_DIR"
        rm -f "$LOCK_FILE"
        return 1
    }
    
    # Start the controller in background
    nohup python3 "$PYTHON_SCRIPT" >> "$LOG_FILE" 2>&1 &
    local python_pid=$!
    
    # Save PID
    echo $python_pid > "$PID_FILE"
    
    # Wait a moment to see if it started successfully
    sleep 3
    
    if kill -0 "$python_pid" 2>/dev/null; then
        log_message "Controller started successfully (PID: $python_pid)"
        rm -f "$LOCK_FILE"
        return 0
    else
        log_message "Controller failed to start"
        rm -f "$PID_FILE" "$LOCK_FILE"
        return 1
    fi
}

# Function to stop the controller gracefully
stop_controller() {
    if ! is_controller_running; then
        log_message "Controller not running"
        return 0
    fi
    
    local pid=$(cat "$PID_FILE")
    log_message "Stopping controller (PID: $pid)..."
    
    # Try graceful shutdown first
    kill -TERM "$pid" 2>/dev/null
    
    # Wait up to 10 seconds for graceful shutdown
    for i in {1..10}; do
        if ! kill -0 "$pid" 2>/dev/null; then
            log_message "Controller stopped gracefully"
            rm -f "$PID_FILE"
            return 0
        fi
        sleep 1
    done
    
    # Force kill if necessary
    log_message "Forcing controller stop..."
    kill -KILL "$pid" 2>/dev/null
    rm -f "$PID_FILE"
    
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
case "${1:-check}" in
    "start")
        start_controller
        ;;
    "stop")
        stop_controller
        ;;
    "restart")
        stop_controller
        sleep 2
        start_controller
        ;;
    "status")
        if is_controller_running; then
            echo "Controller is running (PID: $(cat $PID_FILE))"
            exit 0
        else
            echo "Controller is not running"
            exit 1
        fi
        ;;
    "check"|*)
        # Intelligent check and management
        if should_be_active; then
            if is_controller_running; then
                if ! health_check; then
                    log_message "Health check failed, restarting controller"
                    stop_controller
                    sleep 2
                    start_controller
                fi
            else
                log_message "Controller should be running but isn't, starting..."
                start_controller
            fi
        else
            # Should not be active
            if is_controller_running; then
                log_message "Controller running when it shouldn't be, stopping..."
                stop_controller
            fi
        fi
        ;;
esac

exit 0