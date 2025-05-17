# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains an adaptive system that automatically adjusts screen brightness and audio volume based on environmental conditions. The system:

- Uses computer vision to detect ambient light levels via camera
- Captures audio to detect ambient noise levels
- Dynamically adjusts display brightness and sound volume based on these inputs
- Implements smooth transitions with adaptive algorithms
- Features power efficiency optimization through activity detection

## Code Architecture

The main code is in `autoBrighthness_sound40.py` which implements a `BrightnessController` class with these key components:

1. **Initialization and Configuration**: Sets up camera, audio, and system parameters
2. **Activity Monitoring**: Tracks user activity and adjusts system behavior
3. **Frame Processing**: Captures and analyzes camera frames in a separate thread
4. **Brightness Analysis**: Uses OpenCV and NumPy to calculate ambient light levels
5. **Volume Control**: Captures and analyzes ambient noise levels
6. **Adaptive Adjustment**: Implements smoothing algorithms for gradual transitions

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

To run the application:

```bash
python autoBrighthness_sound40.py
```

## Notes for Developers

- The system uses threading to process frames in parallel
- Adaptive intervals are implemented to reduce resource usage during periods of low change
- The system automatically reduces activity during user inactivity
- There are configurable thresholds for minimum/maximum brightness and volume