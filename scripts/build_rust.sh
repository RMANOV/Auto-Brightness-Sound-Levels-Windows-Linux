#!/bin/bash
# Build script for adaptive-rust with optimizations

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RUST_DIR="$PROJECT_DIR/adaptive-rust"

echo "Building adaptive-rust..."
echo "Project directory: $PROJECT_DIR"
echo "Rust directory: $RUST_DIR"

cd "$RUST_DIR"

# Check if cargo is available
if ! command -v cargo &> /dev/null; then
    echo "Error: cargo not found. Please install Rust."
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

# Build mode
BUILD_MODE="${1:-release}"

case "$BUILD_MODE" in
    release)
        echo "Building in release mode with LTO..."
        RUSTFLAGS="-C target-cpu=native" cargo build --release

        echo ""
        echo "Binary built: $RUST_DIR/target/release/adaptive-controller"
        ls -lh "$RUST_DIR/target/release/adaptive-controller"
        ;;

    dev)
        echo "Building in development mode..."
        cargo build

        echo ""
        echo "Binary built: $RUST_DIR/target/debug/adaptive-controller"
        ;;

    bench)
        echo "Running benchmarks..."
        cargo bench
        ;;

    test)
        echo "Running tests..."
        cargo test
        ;;

    python)
        echo "Building Python bindings with maturin..."

        if ! command -v maturin &> /dev/null; then
            echo "Installing maturin..."
            pip install maturin
        fi

        cd "$RUST_DIR/crates/ffi"
        maturin develop --release

        echo ""
        echo "Python module installed. Test with:"
        echo "  python -c 'import adaptive_rust; print(adaptive_rust.version())'"
        ;;

    *)
        echo "Usage: $0 [release|dev|bench|test|python]"
        echo ""
        echo "  release  - Build optimized release binary (default)"
        echo "  dev      - Build debug binary for development"
        echo "  bench    - Run Criterion benchmarks"
        echo "  test     - Run unit tests"
        echo "  python   - Build Python bindings with maturin"
        exit 1
        ;;
esac

echo ""
echo "Build complete!"
