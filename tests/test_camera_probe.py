import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location("camera_probe", Path(__file__).resolve().parents[1] / "camera_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def sample(value, timestamp=1000):
    return {"schema": 1, "source": probe.SOURCE, "ambient": value, "captured_at": timestamp}


class ProbeTests(unittest.TestCase):
    def cv(self, values, opened=True, bad_frame=False, resize=True):
        class Camera:
            def __init__(self): self.released = False; self.size_requests = []
            def isOpened(self): return opened
            def set(self, *args): self.size_requests.append(args); return resize
            def read(self):
                value = next(values)
                if value is None: return False, None
                return True, SimpleNamespace(size=0 if bad_frame else 1, mean=lambda: value)
            def release(self): self.released = True
        cap = Camera()
        cv = SimpleNamespace(VideoCapture=lambda *args: cap, CAP_V4L2=1,
                             CAP_PROP_FRAME_WIDTH=2, CAP_PROP_FRAME_HEIGHT=3,
                             COLOR_BGR2GRAY=4, cvtColor=lambda frame, mode: frame)
        return cv, cap

    def test_drain_warmup_and_median_then_release_without_persisting(self):
        cv, cap = self.cv(iter([250, 240, 230, 0, 255, 0]))
        self.assertEqual(probe.measure(cv, clock=lambda: 0, wall_clock=lambda: 1000), sample(0))
        self.assertTrue(cap.released)

    def test_unsupported_resize_keeps_valid_negotiated_frames(self):
        cv, cap = self.cv(iter([250, 240, 230, 0, 255, 0]), resize=False)
        self.assertEqual(probe.measure(cv, clock=lambda: 0, wall_clock=lambda: 1000), sample(0))
        self.assertEqual(cap.size_requests, [(cv.CAP_PROP_FRAME_WIDTH, 320),
                                            (cv.CAP_PROP_FRAME_HEIGHT, 240)])
        self.assertTrue(cap.released)

    def test_unsupported_resize_does_not_accept_an_empty_frame(self):
        cv, cap = self.cv(iter([10]), bad_frame=True, resize=False)
        with self.assertRaisesRegex(RuntimeError, "no frame"):
            probe.measure(cv, clock=lambda: 0)
        self.assertTrue(cap.released)

    def test_missing_empty_nonfinite_and_unavailable_camera_are_errors(self):
        for values, opened, empty in [([None], True, False), ([10], True, True),
                                      ([float('nan')], True, False), ([], False, False)]:
            with self.subTest(values=values, opened=opened, empty=empty):
                cv, cap = self.cv(iter(values), opened, empty)
                with self.assertRaises((ValueError, RuntimeError)):
                    probe.measure(cv, clock=lambda: 0)
                self.assertTrue(cap.released)

    def test_deadline_includes_time_consumed_by_read(self):
        cv, cap = self.cv(iter([10]))
        clock = iter([0, 0, 3])
        with self.assertRaises(TimeoutError):
            probe.measure(cv, clock=lambda: next(clock))
        self.assertTrue(cap.released)

    def test_deadline_includes_final_frame_processing(self):
        cv, cap = self.cv(iter([10] * 6))
        state = {"time": 0, "conversions": 0}
        def convert(frame, mode):
            state["conversions"] += 1
            if state["conversions"] == 6:
                state["time"] = 3
            return frame
        cv.cvtColor = convert
        with self.assertRaises(TimeoutError):
            probe.measure(cv, clock=lambda: state["time"])
        self.assertTrue(cap.released)

    def test_same_domain_baseline_and_strict_threshold(self):
        self.assertFalse(probe.changed(sample(60), sample(60, 900)))
        self.assertTrue(probe.changed(sample(60), sample(30, 900)))
        self.assertFalse(probe.changed(sample(42), sample(30, 900)))
        self.assertTrue(probe.changed(sample(42.01), sample(30, 900)))
        self.assertFalse(probe.changed(sample(0), sample(0, 900)))

    def test_invalid_stale_or_screen_state_bootstraps(self):
        for baseline in [None, {}, {"brightness": 34}, sample(60, 1001), sample(60, -90000),
                         {**sample(60), "source": "old-method"}]:
            self.assertTrue(probe.changed(sample(60), baseline))

    def test_failed_measurement_never_replaces_verified_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ambient.json"
            probe.accept(path, sample(30), now=1001)
            before = path.read_bytes()
            for invalid in [sample(float('nan')), sample(30, 2000), sample(30, 0)]:
                with self.assertRaises(ValueError): probe.accept(path, invalid, now=1001)
                self.assertEqual(path.read_bytes(), before)
            self.assertFalse(probe.changed(sample(35, 1100), probe.read_baseline(path)))
            self.assertEqual(path.read_bytes(), before)
            self.assertTrue(probe.changed(sample(43, 1200), probe.read_baseline(path)))

    def test_duplicate_fields_and_bool_schema_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.json"
            path.write_text('{"ambient": 1, "ambient": 99}')
            with self.assertRaises(ValueError): probe.read_json(path)
        with self.assertRaises(ValueError): probe.validate_sample({**sample(30), "schema": True})


if __name__ == "__main__": unittest.main()
