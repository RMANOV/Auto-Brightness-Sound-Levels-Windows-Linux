#!/usr/bin/env python3
"""
Benchmark script to test Numba JIT performance improvements
"""

import numpy as np
import time
from numba import njit

# Import the optimized functions
from adaptive_brightness_volume import AdaptiveBrightnessVolumeController

def benchmark_function(func, *args, iterations=1000, name="function"):
    """Benchmark a function and return average execution time"""
    times = []
    
    # Warmup runs (especially important for JIT functions)
    for _ in range(10):
        func(*args)
    
    # Actual benchmark
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"{name:30s}: {avg_time:8.3f}ms avg ({min_time:6.3f}-{max_time:6.3f}ms)")
    return avg_time

def main():
    print("🚀 Numba JIT Performance Benchmark")
    print("=" * 60)
    
    # Create test data
    test_frame = np.random.randint(0, 255, (240, 320), dtype=np.uint8)
    test_audio = np.random.random(4410).astype(np.float32) * 0.01  # Realistic audio levels
    test_img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    # Camera brightness test values
    test_camera_brightness = [15.0, 45.0, 75.0]
    min_brightness, max_brightness = 5.0, 45.0
    
    # Volume test values
    test_normalized_noise = [0.1, 0.5, 0.8]
    min_volume, max_volume = 3.0, 35.0
    
    print("\n📊 JIT-Compiled Function Performance:")
    print("-" * 60)
    
    # Benchmark JIT-compiled functions
    controller = AdaptiveBrightnessVolumeController()
    
    # Brightness calculation
    avg_brightness = benchmark_function(
        controller.calculate_brightness, 
        test_frame, 
        iterations=5000,
        name="calculate_brightness (JIT)"
    )
    
    # Audio noise level calculation
    avg_audio = benchmark_function(
        controller._compute_noise_level_jit,
        test_audio,
        iterations=5000,
        name="compute_noise_level (JIT)"
    )
    
    # Brightness mapping
    for i, camera_val in enumerate(test_camera_brightness):
        avg_mapping = benchmark_function(
            controller._calculate_brightness_mapping_jit,
            camera_val, min_brightness, max_brightness,
            iterations=10000,
            name=f"brightness_mapping_{i+1} (JIT)"
        )
    
    # Volume mapping
    for i, noise_val in enumerate(test_normalized_noise):
        avg_volume = benchmark_function(
            controller._calculate_volume_mapping_jit,
            noise_val, min_volume, max_volume,
            iterations=10000,
            name=f"volume_mapping_{i+1} (JIT)"
        )
    
    # Smoothing transitions
    avg_smooth = benchmark_function(
        controller._smooth_transition_jit,
        25.0, 30.0, 0.3,
        iterations=10000,
        name="smooth_transition (JIT)"
    )
    
    # Screen brightness analysis
    avg_screen = benchmark_function(
        controller._analyze_screen_brightness_jit,
        test_img,
        iterations=1000,
        name="screen_brightness_analysis (JIT)"
    )
    
    # Change detection
    avg_change = benchmark_function(
        controller._check_significant_change_jit,
        45.0, 30.0, True,
        iterations=10000,
        name="significant_change_check (JIT)"
    )
    
    print("\n🔬 Performance Analysis:")
    print("-" * 60)
    
    # Calculate some performance metrics
    total_core_functions_time = avg_brightness + avg_audio + avg_smooth
    
    print(f"Core functions total time    : {total_core_functions_time:.3f}ms per cycle")
    print(f"Estimated cycles per second  : {1000/total_core_functions_time:.0f}")
    print(f"Main loop efficiency target  : <500ms update interval")
    
    # Performance recommendations
    print(f"\n💡 Optimization Status:")
    print("-" * 60)
    
    if avg_brightness < 0.1:
        print("✅ Brightness calculation: EXCELLENT (< 0.1ms)")
    elif avg_brightness < 0.5:
        print("✅ Brightness calculation: GOOD (< 0.5ms)")
    else:
        print("⚠️  Brightness calculation: Could be improved")
    
    if avg_audio < 0.2:
        print("✅ Audio processing: EXCELLENT (< 0.2ms)")
    elif avg_audio < 1.0:
        print("✅ Audio processing: GOOD (< 1.0ms)")
    else:
        print("⚠️  Audio processing: Could be improved")
    
    if avg_smooth < 0.01:
        print("✅ Smoothing operations: EXCELLENT (< 0.01ms)")
    elif avg_smooth < 0.05:
        print("✅ Smoothing operations: GOOD (< 0.05ms)")
    else:
        print("⚠️  Smoothing operations: Could be improved")
    
    print("\n🎯 Summary:")
    print("-" * 60)
    print("All critical functions have been optimized with Numba JIT compilation.")
    print("Performance improvements should be significant especially after warmup.")
    print("Monitor the real-time performance stats during actual usage for verification.")

if __name__ == "__main__":
    main()