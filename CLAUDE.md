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
- **Efficient Memory Management**: Smart caching and resource cleanup

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

To run the main application:

```bash
python3 adaptive_brightness_volume.py
```

To run performance benchmarks:

```bash
python3 benchmark_numba.py
```

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