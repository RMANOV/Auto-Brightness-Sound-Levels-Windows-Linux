#!/usr/bin/env python3
"""
Launch script for the Rust adaptive brightness/volume controller.

This script launches the optimized Rust binary with proper signal handling
and provides a Python interface for integration with existing scripts.
"""

import signal
import subprocess
import sys
from pathlib import Path

# Find project root
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_DIR = SCRIPT_DIR.parent
RUST_BINARY = PROJECT_DIR / "adaptive-rust" / "target" / "release" / "adaptive-controller"
RUST_BINARY_DEBUG = PROJECT_DIR / "adaptive-rust" / "target" / "debug" / "adaptive-controller"


class RustController:
    """Wrapper for the Rust controller binary."""

    def __init__(self, use_debug: bool = False):
        self.binary = RUST_BINARY_DEBUG if use_debug else RUST_BINARY
        self.process: subprocess.Popen | None = None

    def is_built(self) -> bool:
        """Check if the binary is built."""
        return self.binary.exists()

    def build(self, release: bool = True) -> bool:
        """Build the Rust binary."""
        build_script = SCRIPT_DIR / "build_rust.sh"
        mode = "release" if release else "dev"

        try:
            result = subprocess.run([str(build_script), mode], cwd=PROJECT_DIR, check=True)
            return result.returncode == 0
        except subprocess.CalledProcessError:
            return False

    def start(self) -> bool:
        """Start the controller."""
        if not self.is_built():
            print(f"Binary not found at {self.binary}")
            print("Building...")
            if not self.build():
                print("Build failed!")
                return False

        print(f"Starting Rust controller: {self.binary}")

        try:
            self.process = subprocess.Popen(
                [str(self.binary)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )

            # Read initial output
            for _ in range(10):
                if self.process.poll() is not None:
                    # Process exited
                    break

                try:
                    line = self.process.stdout.readline()
                    if line:
                        print(line.rstrip())
                except Exception:
                    break

            return self.process.poll() is None

        except Exception as e:
            print(f"Failed to start controller: {e}")
            return False

    def stop(self) -> None:
        """Stop the controller gracefully."""
        if self.process and self.process.poll() is None:
            print("Stopping controller...")
            self.process.send_signal(signal.SIGTERM)

            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("Force killing controller...")
                self.process.kill()
                self.process.wait()

    def run_foreground(self) -> int:
        """Run controller in foreground with output streaming."""
        if not self.is_built():
            print(f"Binary not found at {self.binary}")
            print("Building...")
            if not self.build():
                print("Build failed!")
                return 1

        print(f"Running Rust controller: {self.binary}")
        print("Press Ctrl+C to stop")
        print()

        try:
            result = subprocess.run([str(self.binary)])
            return result.returncode
        except KeyboardInterrupt:
            return 0

    @property
    def is_running(self) -> bool:
        """Check if controller is running."""
        return self.process is not None and self.process.poll() is None


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Launch the Rust adaptive brightness/volume controller")
    parser.add_argument("--debug", action="store_true", help="Use debug build instead of release")
    parser.add_argument("--build-only", action="store_true", help="Only build, don't run")
    parser.add_argument("--background", action="store_true", help="Run in background")

    args = parser.parse_args()

    controller = RustController(use_debug=args.debug)

    if args.build_only:
        success = controller.build(release=not args.debug)
        return 0 if success else 1

    if args.background:
        if controller.start():
            print(f"Controller started (PID: {controller.process.pid})")
            return 0
        else:
            return 1
    else:
        return controller.run_foreground()


if __name__ == "__main__":
    sys.exit(main())
