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

# Function to perform simplified flash detection for significant changes
flash_detection_check() {
    # Simple approach: 50-second wait (35s warmup + 15s buffer) then compare with saved state
    local temp_log="/tmp/flash_detection.log"
    
    # Get current saved state
    local saved_brightness=30
    local saved_volume=20
    
    if [[ -f "/home/rmanov/.config/adaptive-controller/last_state.txt" ]]; then
        saved_brightness=$(grep "brightness=" "/home/rmanov/.config/adaptive-controller/last_state.txt" | cut -d'=' -f2 2>/dev/null || echo 30)
        saved_volume=$(grep "volume=" "/home/rmanov/.config/adaptive-controller/last_state.txt" | cut -d'=' -f2 2>/dev/null || echo 20)
    fi
    
    log_message "Flash detection: Starting simplified 50-second detection (35s warmup + 15s buffer)"
    
    # Simple environmental sampling with generous buffer
    cd "$SCRIPT_DIR"
    timeout 60s python3 -c "
import sys
sys.path.append('$SCRIPT_DIR')
import time
import cv2
import numpy as np

try:
    print('flash_status:starting_simplified_detection')
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print('flash_error:camera_not_available')
        sys.exit(1)
    
    # Configure camera quickly
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    # Wait full 50 seconds (35s warmup + 15s buffer for slow systems)
    print('flash_status:waiting_50_seconds_for_stability')
    time.sleep(50)
    
    # Take simple measurement after full wait
    print('flash_status:taking_final_measurement')
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_brightness = np.mean(gray) / 255 * 100
        
        # Calculate percentage change from saved state
        brightness_change = abs(current_brightness - $saved_brightness) / max($saved_brightness, 1) * 100
        
        print(f'flash_brightness:{current_brightness:.1f}')
        print(f'flash_change:{brightness_change:.1f}')
        print(f'flash_saved:$saved_brightness')
    else:
        print('flash_error:no_measurement_possible')
        
    cap.release()
        
except Exception as e:
    print(f'flash_error:{e}')
" > "$temp_log" 2>&1
    
    # Parse results
    if [[ -f "$temp_log" ]]; then
        local brightness_change=$(grep "flash_change:" "$temp_log" | cut -d':' -f2 2>/dev/null || echo 0)
        local current_brightness=$(grep "flash_brightness:" "$temp_log" | cut -d':' -f2 2>/dev/null || echo 0)
        
        # Check if change is significant (>40%)
        if (( $(echo "$brightness_change > 40.0" | bc -l 2>/dev/null || echo 0) )); then
            log_message "Flash detection: Significant change detected - Current: ${current_brightness}%, Saved: ${saved_brightness}%, Change: ${brightness_change}%"
            rm -f "$temp_log"
            return 0  # Significant change - should activate
        else
            log_message "Flash detection: No significant change - Current: ${current_brightness}%, Saved: ${saved_brightness}%, Change: ${brightness_change}% (threshold: 40%)"
            rm -f "$temp_log"
            return 1  # No significant change - skip activation
        fi
    fi
    
    rm -f "$temp_log"
    log_message "Flash detection: Failed to get measurements - defaulting to skip"
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

# Function to start the controller for optimized burst mode
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
    
    log_message "Starting adaptive controller in optimized burst mode..."
    
    # Change to script directory
    cd "$SCRIPT_DIR" || {
        log_message "Failed to change to script directory: $SCRIPT_DIR"
        rm -f "$LOCK_FILE"
        return 1
    }
    
    # Start controller with timeout for burst mode
    # Run for 8 minutes (enough for full warmup + adjustments + stabilization)
    timeout 480s python3 "$PYTHON_SCRIPT" >> "$LOG_FILE" 2>&1 &
    local python_pid=$!
    
    # Save PID
    echo $python_pid > "$PID_FILE"
    
    # Wait a moment to see if it started successfully
    sleep 3
    
    if kill -0 "$python_pid" 2>/dev/null; then
        log_message "Controller started successfully in burst mode (PID: $python_pid, 8min timeout)"
        rm -f "$LOCK_FILE"
        
        # Set up automatic cleanup after burst mode
        (
            # Wait for either timeout or normal completion
            wait $python_pid 2>/dev/null
            local exit_code=$?
            
            # Clean up PID file
            if [[ -f "$PID_FILE" ]]; then
                local stored_pid=$(cat "$PID_FILE")
                if [[ "$stored_pid" == "$python_pid" ]]; then
                    rm -f "$PID_FILE"
                fi
            fi
            
            if [[ $exit_code -eq 124 ]]; then
                log_message "Controller completed burst mode successfully (8min timeout)"
            else
                log_message "Controller completed burst mode (exit code: $exit_code)"
            fi
        ) &
        
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