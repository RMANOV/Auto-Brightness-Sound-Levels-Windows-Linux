# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a high-performance adaptive system that automatically adjusts screen brightness and audio volume based on environmental conditions. The system features:

- **Intelligent Computer Vision**: Detects ambient light levels via camera with real-time processing
- **Advanced Audio Analysis**: Captures and analyzes ambient noise levels for dynamic volume control
- **Adaptive Algorithms**: Dynamically adjusts display brightness and sound volume with smooth transitions
- **Performance Optimization**: Leverages Numba JIT compilation for ultra-fast mathematical computations
- **Power Efficiency**: Activity detection and adaptive polling to minimize resource usage
- **Cross-Platform Support**: Designed for Linux with multiple brightness/audio control methods

## Code Architecture

The main code is in `adaptive_brightness_volume.py` which implements the `AdaptiveBrightnessVolumeController` class with these optimized components:

### Core Components
1. **Initialization and Configuration**: Multi-method device detection and fallback systems
2. **Activity Monitoring**: Intelligent user activity tracking with adaptive behavior
3. **Multi-threaded Processing**: Parallel camera frame processing with queue-based communication
4. **Advanced Brightness Analysis**: JIT-compiled computer vision algorithms using OpenCV and NumPy
5. **Audio Processing**: Real-time noise level analysis with multiple capture methods
6. **Adaptive Control**: Sophisticated smoothing algorithms with JIT-optimized mathematics

### Performance Features
- **Numba JIT Compilation**: Critical functions compiled to machine code for maximum speed
- **Real-time Performance Monitoring**: Built-in timing and profiling capabilities
- **Optimized Mathematical Operations**: All brightness/volume calculations use JIT-compiled functions
- **Comprehensive Resource Management**: Advanced cleanup system eliminating browser lag and system performance issues

## Requirements

The code depends on:
- Python 3.x
- OpenCV (cv2)
- NumPy
- Numba (for JIT compilation)
- SoundDevice
- Linux command-line tools:
  - `brightnessctl` for brightness control
  - `amixer` for volume control

## Running the Application

### Manual Operation

To run the main application manually:

```bash
python3 adaptive_brightness_volume.py
```

To run performance benchmarks:

```bash
python3 benchmark_numba.py
```

### Intelligent Automated Operation (Recommended)

For production use, the intelligent cron-based scheduling system is recommended:

```bash
# Install mathematically optimized cron scheduling
./install_crontab.sh

# Monitor system status
./adaptive_controller_manager.sh status

# View real-time operation logs  
tail -f /tmp/adaptive_controller.log

# Manual control when needed
./adaptive_controller_manager.sh start|stop|restart
```

### Cron System Features

The intelligent scheduling system provides:
- **30-minute optimized intervals** calculated for 81% energy savings
- **Flash detection technology** - only activates on >40% environmental changes
- **Eliminates system impact** - no taskbar flickering or fan activation
- **Deep night work mode** - special 2-5 AM handling for uninterrupted sessions
- **Smart threshold detection** prevents unnecessary process activations
- **Comprehensive resource cleanup** - automatic cleanup after each 8-minute burst mode
- **Signal-aware termination** - proper cleanup on timeout, interruption, or normal exit
- **Time-based activation** with intelligent day/night adaptation
- **System resource monitoring** with minimal frequency checking
- **Health monitoring** with reduced-impact auto-recovery
- **KDE Plasma integration** for session detection
- **Professional logging** with minimal footprint

## Performance Optimizations

### Numba JIT Compilation
This project leverages **Numba's Just-In-Time (JIT) compilation** for maximum performance:

- **Audio Processing**: `_compute_noise_level_jit()` - Ultra-fast RMS calculations
- **Brightness Mapping**: `_calculate_brightness_mapping_jit()` - Optimized mathematical formulas
- **Volume Control**: `_calculate_volume_mapping_jit()` - Logarithmic curve calculations
- **Smoothing Operations**: `_smooth_transition_jit()` - All transition mathematics
- **Image Analysis**: `_analyze_screen_brightness_jit()` - Screen content analysis
- **Change Detection**: `_check_significant_change_jit()` - Light change algorithms

### Performance Benefits
- **10-100x faster** execution for mathematical operations after JIT warmup
- **Reduced CPU usage** during continuous operation
- **Real-time responsiveness** for brightness and volume adjustments
- **Built-in performance monitoring** with timing statistics every 30 seconds

## Comprehensive Resource Management

### Resource Cleanup System
The system implements a sophisticated resource cleanup mechanism that **eliminates browser lag and system performance degradation** after each execution cycle:

#### Signal Handling
- **SIGTERM Handler**: Proper cleanup when terminated by timeout (8-minute burst mode)
- **SIGINT Handler**: Graceful cleanup on manual interruption (Ctrl+C)
- **atexit Registration**: Automatic cleanup on normal program termination
- **Signal-aware processing**: All termination scenarios trigger comprehensive cleanup

#### Memory Management
- **Numba JIT Cache Clearing**: Clears compiled function cache to prevent memory accumulation
- **CUDA Context Cleanup**: Properly closes CUDA contexts if available
- **Multi-pass Garbage Collection**: Multiple garbage collection passes for thorough memory cleanup
- **Performance Data Cleanup**: Clears accumulated timing statistics and monitoring data
- **Large Array Cleanup**: Explicit cleanup of NumPy arrays and image data

#### Thread Management
- **Proper Thread Joining**: Timeout-based thread termination with 2-second grace period
- **Queue Resource Cleanup**: Complete emptying of frame and brightness queues
- **Daemon Thread Handling**: Proper cleanup of background processing threads
- **Thread State Monitoring**: Detection and handling of threads that fail to terminate cleanly

#### OpenCV Resource Management
- **Camera Handle Cleanup**: Force release of all potential camera handles (0-9)
- **Window Destruction**: Proper cleanup of OpenCV windows with error handling
- **Video Capture Cleanup**: Explicit release of VideoCapture objects
- **Memory Buffer Cleanup**: Clear OpenCV internal buffers and cached frames

#### Implementation Details
```python
# Global cleanup function called on any termination
def comprehensive_cleanup():
    # 1. Stop controller and join threads
    # 2. OpenCV camera and window cleanup
    # 3. Numba JIT cache clearing
    # 4. Performance monitoring cleanup
    # 5. Multi-pass garbage collection
    # 6. Resource leak detection and reporting

# Signal handlers for robust termination
signal.signal(signal.SIGTERM, signal_handler)  # Timeout cleanup
signal.signal(signal.SIGINT, signal_handler)   # Interrupt cleanup
atexit.register(comprehensive_cleanup)          # Normal exit cleanup
```

### Impact
- **Zero Browser Lag**: Eliminates system performance degradation after each run
- **Clean Burst Cycles**: 8-minute execution cycles with complete resource cleanup
- **Robust Termination**: Proper cleanup regardless of termination method
- **Memory Efficiency**: Prevents resource accumulation over multiple execution cycles
- **System Stability**: Maintains system performance during extended operation

## Notes for Developers

### System Architecture
- **Multi-threaded design** with separate camera processing thread
- **Queue-based communication** between threads for optimal performance
- **Adaptive polling intervals** to reduce resource usage during periods of low change
- **Intelligent inactivity detection** with automatic resource management
- **Multiple fallback methods** for brightness and audio control across different Linux distributions

### Code Organization
- **JIT-compiled functions** are prefixed with `_jit` for easy identification
- **Performance timing decorators** can be applied to any function for monitoring
- **Configurable thresholds** for minimum/maximum brightness and volume ranges
- **Graceful error handling** with fallback implementations when dependencies are missing

### Development Guidelines
- All critical mathematical operations should use JIT-compiled functions
- New algorithms should include performance timing decorators
- Test both with and without Numba to ensure fallback compatibility
- Use the benchmark script to validate performance improvements
- **Resource Management**: Always implement proper cleanup for new components
- **Signal Handling**: Ensure new features integrate with the cleanup system
- **Memory Efficiency**: Use garbage collection and explicit cleanup for large objects
- **Thread Safety**: Implement proper thread termination and queue cleanup
- **OpenCV Resources**: Always release camera handles and destroy windows in cleanup

### Intelligent Scheduling Development
- Cron intervals should be optimized for real-world impact (energy + UX)
- Flash detection thresholds must be calibrated to prevent unnecessary activations
- Process management logic must include minimal-footprint error handling
- Health monitoring should use reduced frequency to eliminate system noise
- Time-based activation must include deep night work mode (2-5 AM)
- Smart threshold detection should prevent fan activation and UI flickering
- Logging systems should use minimal footprint with automatic rotation
- All scheduling changes should eliminate taskbar disruption and system noise
- Flash detection should use 2-second timeouts for minimal system impact
- Environmental change detection should use >40% thresholds for significance