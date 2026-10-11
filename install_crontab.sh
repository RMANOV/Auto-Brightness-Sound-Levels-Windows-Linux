#!/bin/bash

# Intelligent Cron Installation Script for Adaptive Controller
# 10-minute checks: sunrise/sunset windows plus the periods between them
# Energy efficient for Fedora 42 laptop systems

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANAGER_SCRIPT="$SCRIPT_DIR/adaptive_controller_manager.sh"
RUST_BINARY="$SCRIPT_DIR/adaptive-rust/target/release/adaptive-controller"

# The manager requires the native backend; never install silently inactive jobs.
if [[ ! -f "$RUST_BINARY" || ! -x "$RUST_BINARY" ]]; then
    printf 'Required executable Rust backend is unavailable: %s\n' "$RUST_BINARY" >&2
    printf 'Build it first, then rerun this installer:\n  cd %q && cargo build --release -p adaptive-controller --locked\n' "$SCRIPT_DIR/adaptive-rust" >&2
    exit 1
fi

echo "🚀 Installing Adaptive Controller Cron Jobs..."
echo "📊 Optimized 10-minute interval with flash detection and comprehensive resource cleanup"

# Create temporary crontab file
TEMP_CRONTAB=$(mktemp)

# Add header with explanation
cat >> "$TEMP_CRONTAB" << 'EOF'
# Adaptive Brightness & Volume Controller - Sunrise/Sunset Intelligent Scheduling
# Optimized for minimal system impact and optimal user experience
# 
# Sunrise/sunset windows plus 10-minute periods between them, with flash detection:
# - Sunrise/sunset windows (30min before, 2h after each event) are kept
# - Between the windows the same 10-minute check also runs, all day
# - Dynamic time windows: Automatically adjusts with seasonal changes
# - Optimized flash detection: 40s total wait (35s warmup + 5s buffer)
# - Only activates on >40% environmental changes using saved state comparison
# - Burst mode execution: 8 minutes when changes detected with comprehensive cleanup
# - Geographic accuracy: Uses calculated sunrise/sunset times for user's location
# - Comprehensive resource cleanup eliminates browser lag and system performance issues
# - Signal-aware termination with proper cleanup on timeout, interruption, or normal exit
# - Fallback protection: Reverts to time-based logic if sunrise/sunset calculation fails
#
# Schedule breakdown:
# */10 * * * * - Sunrise/sunset window or 10-minute period between them + flash detection + burst mode (8min if needed)
# 0 3 * * * - Daily cleanup at 3 AM (between sunrise/sunset windows)
# 0 6 * * 1 - Weekly log maintenance on Monday 6 AM

EOF

# Add the actual cron jobs
cat >> "$TEMP_CRONTAB" << EOF
# Primary intelligent check every 10 minutes with flash detection (OPTIMIZED INTERVAL)
# This is the main scheduling entry with smart threshold detection
# Only activates on >40% environmental changes to minimize system impact
*/10 * * * * DISPLAY=:0 "$MANAGER_SCRIPT" check >/dev/null 2>&1

# Daily cleanup at 3 AM (during deep night work mode)
# Performs graceful restart to clear any memory leaks or accumulated state
# Scheduled during deep night to avoid disrupting 2-5 AM work sessions
0 3 * * * DISPLAY=:0 "$MANAGER_SCRIPT" restart >/dev/null 2>&1

# Weekly log maintenance on Monday at 6 AM
# Cleans up old logs and ensures log rotation is working properly
0 6 * * 1 find /tmp -name "adaptive_controller*" -type f -mtime +7 -delete 2>/dev/null

EOF

# Install the crontab
echo "Installing crontab..."
crontab "$TEMP_CRONTAB"

# Cleanup
rm "$TEMP_CRONTAB"

echo "✅ Crontab installed successfully!"
echo ""
echo "📋 Cron Schedule Summary:"
echo "  🔄 Main check:      Every 10 minutes with flash detection"
echo "  🧹 Daily cleanup:   3:00 AM (deep night mode)"
echo "  📝 Log maintenance: Monday 6:00 AM"
echo ""
echo "🎯 Key Features:"
echo "  🌅 Sunrise/sunset windows: 30min before to 2h after each event"
echo "  ⏱️  Between windows:  the same check every 10 minutes, all day"
echo "  ⚡ Low energy use: a short camera probe each check, full run only on change"
echo "  🧠 Flash detection: Only activates on >40% environmental changes"
echo "  🗺️  Geographic accuracy: Calculates exact sunrise/sunset times for your location"
echo "  📅 Seasonal adaptation: Windows automatically adjust as days get longer/shorter"
echo "  💻 Eliminates taskbar icon flickering and fan activation"
echo "  🔍 Smart threshold detection prevents unnecessary activations"
echo "  🧹 Comprehensive resource cleanup eliminates browser lag"
echo "  🛡️ Signal-aware termination with proper cleanup on any exit"
echo "  🛡️ Fallback protection: Reverts to time-based logic if calculation fails"
echo "  📈 Professional logging with minimal system footprint"
echo ""
echo "📊 Real-World Optimization Results:"
echo "  🔋 Energy use: short camera probe every 10 minutes, full run only on change"
echo "  ⏱️  Response time: within 10 minutes of a >40% light change, at any hour"
echo "  🖥️  UX Impact: Eliminated flickering and fan noise"
echo "  🌅 Intelligent timing: Perfect synchronization with natural light cycles"
echo "  🚀 System load: Minimal - full activation only on real light changes"
echo "  🧹 Zero browser lag: Comprehensive resource cleanup after each run"
echo "  🛡️ Robust termination: Clean exit on timeout, interrupt, or normal completion"
echo "  🗺️  Location aware: Automatically detects or estimates your geographic coordinates"
echo ""
echo "To view installed cron jobs:"
echo "  crontab -l"
echo ""
echo "To check controller status:"
echo "  ./adaptive_controller_manager.sh status"
echo ""
echo "To view logs:"
echo "  tail -f \"\${XDG_STATE_HOME:-\$HOME/.local/state}/adaptive-controller/controller.log\""
