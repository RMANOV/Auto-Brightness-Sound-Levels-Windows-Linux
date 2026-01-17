//! Criterion benchmarks for adaptive-core
//!
//! Run with: cargo bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use adaptive_core::*;

fn bench_compute_noise_level(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_noise_level");

    for size in [4410, 44100, 441000] {
        let audio: Vec<f32> = (0..size)
            .map(|i| (i as f32 * 0.001).sin())
            .collect();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("sequential", size),
            &audio,
            |b, audio| {
                b.iter(|| compute_noise_level(black_box(audio)))
            },
        );
    }

    group.finish();
}

fn bench_calculate_brightness(c: &mut Criterion) {
    let mut group = c.benchmark_group("calculate_brightness");

    for size in [320 * 240, 640 * 480, 1920 * 1080] {
        let frame: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::new("frame", size),
            &frame,
            |b, frame| {
                b.iter(|| calculate_brightness(black_box(frame)))
            },
        );
    }

    group.finish();
}

fn bench_brightness_mapping(c: &mut Criterion) {
    let mut group = c.benchmark_group("brightness_mapping");

    group.bench_function("standard", |b| {
        b.iter(|| {
            calculate_brightness_mapping(
                black_box(45.0),
                black_box(5.0),
                black_box(45.0),
            )
        })
    });

    group.bench_function("branchless", |b| {
        b.iter(|| {
            adaptive_core::brightness::calculate_brightness_mapping_branchless(
                black_box(45.0),
                black_box(5.0),
                black_box(45.0),
            )
        })
    });

    group.finish();
}

fn bench_volume_mapping(c: &mut Criterion) {
    let mut group = c.benchmark_group("volume_mapping");

    group.bench_function("standard", |b| {
        b.iter(|| {
            calculate_volume_mapping(
                black_box(0.5),
                black_box(3.0),
                black_box(35.0),
            )
        })
    });

    group.bench_function("fast_log", |b| {
        b.iter(|| {
            adaptive_core::volume::calculate_volume_mapping_fast(
                black_box(0.5),
                black_box(3.0),
                black_box(35.0),
            )
        })
    });

    group.finish();
}

fn bench_smooth_transition(c: &mut Criterion) {
    c.bench_function("smooth_transition", |b| {
        b.iter(|| {
            smooth_transition(
                black_box(10.0),
                black_box(20.0),
                black_box(0.3),
            )
        })
    });
}

fn bench_analyze_screen(c: &mut Criterion) {
    let mut group = c.benchmark_group("analyze_screen");

    for size in [960 * 540, 1920 * 1080] {
        let pixels: Vec<u8> = (0..size).map(|i| ((i * 7) % 256) as u8).collect();

        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::new("grayscale", size),
            &pixels,
            |b, pixels| {
                b.iter(|| analyze_screen_brightness(black_box(pixels)))
            },
        );
    }

    group.finish();
}

fn bench_check_change(c: &mut Criterion) {
    let mut group = c.benchmark_group("check_significant_change");

    group.bench_function("branched", |b| {
        b.iter(|| {
            check_significant_change(
                black_box(30.0),
                black_box(50.0),
                black_box(true),
            )
        })
    });

    group.bench_function("branchless", |b| {
        b.iter(|| {
            adaptive_core::change::check_significant_change_branchless(
                black_box(30.0),
                black_box(50.0),
                black_box(true),
            )
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_compute_noise_level,
    bench_calculate_brightness,
    bench_brightness_mapping,
    bench_volume_mapping,
    bench_smooth_transition,
    bench_analyze_screen,
    bench_check_change,
);

criterion_main!(benches);
