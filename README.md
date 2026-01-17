# 🚀 Adaptive Brightness & Volume Controller

## High-Performance Display and Audio Management System

An innovative cross-platform system that revolutionizes device display and audio management through intelligent environmental sensing, real-time adjustments, and **blazing-fast performance**.

---

## 🦀 NEW: Rust-Powered Performance Engine

<table>
<tr>
<td width="50%">

### ⚡ Performance Comparison

| Metric | Python+Numba | **Rust** | Gain |
|--------|-------------|----------|------|
| Cycle Time | 12.3ms | **3-6ms** | **2-4x** |
| Memory | 50-80MB | **10-20MB** | **4x** |
| Startup | 2-3s | **<100ms** | **30x** |
| CPU Idle | ~1% | **<0.5%** | **2x** |

</td>
<td width="50%">

### 🎯 Rust Features
- **SIMD-optimized** computations (8-wide vectorization)
- **Zero-copy** NumPy interop via PyO3
- **Lock-free** channel architecture
- **Branchless** change detection
- **Fast log10** approximation

</td>
</tr>
</table>

```bash
# Build & run Rust backend (optional - auto-detected)
./scripts/build_rust.sh release
./scripts/build_rust.sh python   # Python bindings
```

---
## 🎯 Quick Start

### Manual Operation
```bash
# Install dependencies
pip install opencv-python numpy numba sounddevice

# Run the system manually
python3 adaptive_brightness_volume.py

# Run performance benchmarks
python3 benchmark_numba.py
```

### ⏰ Intelligent Automated Scheduling (Recommended)
```bash
# Install mathematically optimized cron scheduling
./install_crontab.sh

# Check controller status
./adaptive_controller_manager.sh status

# View real-time logs
tail -f /tmp/adaptive_controller.log

# Manual control (if needed)
./adaptive_controller_manager.sh start|stop|restart
```

**🌅 Sunrise/Sunset Optimization:** The system uses **intelligent sunrise/sunset scheduling with geographic awareness** for optimal energy efficiency:
- **~90% energy savings** vs continuous running - only activates during light transition periods!
- **Dynamic activation windows** - 30min before to 2h after each sunrise/sunset (automatically calculated for your location)
- **Geographic intelligence** - auto-detects coordinates or uses manual configuration
- **Seasonal adaptation** - windows automatically adjust as days get longer/shorter
- **Optimized flash detection** - 40-second wait then compares with saved state (>40% change threshold)
- **Eliminates unnecessary activations** outside natural light transition periods
- **Fallback protection** - reverts to time-based logic if sunrise/sunset calculation fails

## 🔥 Technical Achievements

✅ **🦀 Rewrote core engine in Rust** with SIMD vectorization for 2-4x speedup over Numba JIT
✅ **Engineered a self-learning controller** utilizing computer vision and audio processing
✅ **Implemented dual-backend architecture** - Rust for max performance, Python+Numba fallback
✅ **Developed predictive algorithms** for seamless transitions with adaptive smoothing  
✅ **Created intelligent activity detection** and power efficiency optimization  
✅ **Achieved real-time performance monitoring** with built-in timing statistics  
✅ **Designed sunrise/sunset scheduling system** with ~90% energy optimization and geographic intelligence  
✅ **Built intelligent process management** with smart threshold detection and auto-recovery  
✅ **Implemented flash detection system** to eliminate unnecessary activations and system noise  
✅ **Created pure Python astronomical calculator** using NOAA algorithms for location-aware time windows  
✅ **Engineered seasonal adaptation system** with dynamic time windows that adjust with changing daylight  
✅ **Engineered comprehensive resource cleanup** eliminating browser lag and system performance issues  
✅ **Developed robust signal handling** with proper SIGTERM/SIGINT cleanup for burst mode termination  
✅ **Implemented advanced memory management** with Numba JIT cache clearing and multi-pass garbage collection  
✅ **Created intelligent thread management** with timeout-based cleanup and queue resource management  
✅ **Ensured cross-platform compatibility** with multiple fallback methods for Linux distributions

🔹 Key Features:

Intelligent Brightness Control:
Real-time ambient light analysis using OpenCV
Dynamic adjustment with predictive algorithms
Smooth transitions preventing eye strain
Auto-calibration based on usage patterns
Smart Audio Management:
Ambient noise level monitoring
Dynamic volume adjustment
Adaptive audio scaling
Noise filtering and normalization
⚡ **Performance Optimization**:
- **🦀 Rust SIMD Engine**: 8-wide vectorized computations for maximum throughput
- **Zero-Copy Interop**: PyO3 bindings with direct NumPy array access
- **Lock-Free Channels**: Crossbeam-based concurrent architecture
- **Branchless Algorithms**: Conditional-move optimized change detection
- **Numba JIT Fallback**: Pure Python backend with 10-100x speedup when Rust unavailable
- **Multi-threaded processing** for minimal system impact
- **Adaptive polling intervals** to reduce resource usage

🧹 **Advanced Resource Management**:
- **Comprehensive Cleanup System**: Eliminates browser lag and system performance degradation
- **Signal Handling**: Robust SIGTERM/SIGINT cleanup for proper burst mode termination
- **Memory Management**: Numba JIT cache clearing with multi-pass garbage collection
- **Thread Management**: Timeout-based cleanup with proper thread joining and queue cleanup
- **OpenCV Resource Cleanup**: Force release of all camera handles and window destruction
- **Performance Data Cleanup**: Automatic clearing of monitoring data and accumulated state
- **Zero Resource Leaks**: Complete cleanup on any termination scenario (normal, timeout, interrupt)

🌅 **Sunrise/Sunset Intelligent Scheduling System**:
- **Geographic Time Windows**: Only activates during 30min before to 2h after sunrise/sunset periods
- **~90% Energy Savings**: Dramatic improvement vs continuous running (up from 81%!)
- **NOAA Astronomical Calculations**: Pure Python implementation for accurate sunrise/sunset times
- **Location Auto-Detection**: Automatically detects coordinates from timezone or manual config
- **Seasonal Adaptation**: Windows automatically adjust as days get longer/shorter throughout year
- **Optimized Flash Detection**: 40-second wait then compares with saved state values (>40% threshold)
- **Eliminates Unnecessary Activations**: Complete deactivation outside natural light transition periods
- **Fallback Protection**: Reverts to time-based logic if astronomical calculations fail
- **Comprehensive Resource Cleanup**: Automatic cleanup after each 8-minute burst mode
- **Signal-Aware Termination**: Proper cleanup on timeout, interruption, or normal exit
- **System Resource Monitoring**: CPU load and memory usage protection
- **Health Monitoring**: Automatic process recovery with minimal frequency
- **KDE Plasma Integration**: User session and display state detection
- **Professional Logging**: Automatic log rotation with minimal footprint

System Integration:

Hardware-agnostic implementation
Native OS controls integration
Robust error handling
Seamless background operation

🔹 **Technical Stack**:

| Layer | Technology | Purpose |
|-------|------------|---------|
| 🦀 **Core Engine** | Rust + SIMD | Ultra-fast computations |
| 🐍 **Bindings** | PyO3 + NumPy | Zero-copy Python interop |
| 📷 **Vision** | OpenCV | Ambient light detection |
| 🔊 **Audio** | cpal / SoundDevice | Noise level analysis |
| ⚡ **Fallback** | Numba JIT | Python-only performance |
| 🔄 **Concurrency** | Crossbeam / Threading | Lock-free parallelism |

## 📊 Performance Benchmarks

```bash
# Rust benchmarks (Criterion)
./scripts/build_rust.sh bench

# Python/Numba benchmarks
python3 benchmark_numba.py
```

### 🏎️ Benchmark Results

| Function | Numba | Rust | Speedup |
|----------|-------|------|---------|
| `compute_noise_level` | 0.15ms | **0.03ms** | **5x** |
| `calculate_brightness` | 0.08ms | **0.02ms** | **4x** |
| `brightness_mapping` | 0.008ms | **0.002ms** | **4x** |
| `volume_mapping` | 0.015ms | **0.003ms** | **5x** |
| `smooth_transition` | 0.0005ms | **0.0001ms** | **5x** |
| `screen_analysis` | 0.08ms | **0.02ms** | **4x** |
| `change_detection` | 0.01ms | **0.002ms** | **5x** |

> 💡 **Tip**: System auto-detects and uses Rust backend when available, with seamless Python fallback.

## 🎯 Impact

🦀 **Rust-powered performance** - 2-4x faster than Numba JIT, 4x lower memory footprint
✨ **Significantly improved user comfort** and productivity through intelligent adaptation
🌅 **~90% energy savings** through sunrise/sunset scheduling with geographic intelligence (improved from 81%!)  
🗺️ **Location-aware operation** with automatic coordinate detection and seasonal adaptation  
🤖 **Autonomous operation** with reliable saved state comparison and minimal system impact  
🖥️ **Enhanced display and audio experience** with eliminated taskbar flickering and fan noise  
💻 **Professional system integration** synchronized with natural light cycles  
📱 **Superior battery life optimization** through precise light transition period targeting  
🌞 **Perfect timing synchronization** with sunrise and sunset events for optimal UX  
🔍 **Intelligent activation filtering** - only operates during meaningful light change periods  
🧹 **Zero system lag** through comprehensive resource cleanup eliminating browser and performance degradation  
🛡️ **Robust termination handling** with signal-aware cleanup preventing resource leaks on any exit scenario  
🚀 **Clean burst cycles** with proper thread, memory, and camera resource management  
📅 **Seasonal intelligence** with automatic daylight adjustment throughout the year

This project demonstrates expertise in computer vision, signal processing, multi-threaded programming, and system optimization, while delivering a practical solution for everyday computing needs.

#Rust #SIMD #PyO3 #ZeroCopy #LockFree #Crossbeam #ComputerVision #Python #NumbaJIT #PerformanceOptimization #SystemOptimization #SoftwareEngineering #Innovation #AdaptiveSystems #CrossPlatform #RealTimeProcessing #AudioProcessing #CronScheduling #EnergyOptimization #BatteryLife #IntelligentAutomation #SunriseSunset #GeographicIntelligence #SeasonalAdaptation #ResourceManagement #MemoryOptimization #SignalHandling #ZeroLag #HighPerformance #SystemsProgramming
