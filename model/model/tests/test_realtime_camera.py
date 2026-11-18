"""최신 프레임 전달, 입력 중단 및 연속 분석 실행을 검증합니다."""
import queue
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch, Mock

import cv2
import numpy as np

from scripts.video_source import LatestFrameCapture
from scripts import demo_camera


class FakeCapture:
    def __init__(self):
        self.frames = queue.Queue()
        self.released = threading.Event()
        self.release_thread = None

    def read(self):
        try:
            item = self.frames.get(timeout=.02)
        except queue.Empty:
            return False, None
        if isinstance(item, Exception):
            raise item
        return True, item

    def release(self):
        self.release_thread = threading.current_thread().name
        self.released.set()


class LatestFrameTests(unittest.TestCase):
    def make_capture(self, timeout=1.0):
        cap = FakeCapture()
        live = LatestFrameCapture(cap, np.zeros((8, 8, 3), dtype=np.uint8), timeout=timeout)
        self.addCleanup(live.release)
        return cap, live

    def wait_until(self, predicate):
        deadline = time.monotonic() + 1
        while not predicate():
            if time.monotonic() >= deadline:
                self.fail('capture worker did not complete')
            time.sleep(.005)

    def test_slow_consumer_gets_newest_frame_without_duplicates(self):
        cap, live = self.make_capture()
        for value in range(1, 11):
            cap.frames.put(np.full((8, 8, 3), value, dtype=np.uint8))
        self.wait_until(lambda: live.latest().sequence == 11)
        packet = live.latest()
        self.assertEqual(int(packet.image[0, 0, 0]), 10)
        self.assertIsNone(live.latest(packet.sequence))

    def test_missing_frames_expire_instead_of_reusing_old_result(self):
        cap, live = self.make_capture(timeout=.06)
        time.sleep(.09)
        with self.assertRaises(RuntimeError):
            live.latest()
        self.assertTrue(cap.released.wait(.5))

    def test_capture_error_is_reported_and_device_released(self):
        cap, live = self.make_capture()
        cap.frames.put(cv2.error('device disconnected'))
        self.assertTrue(cap.released.wait(.5))
        with self.assertRaisesRegex(RuntimeError, 'device disconnected'):
            live.latest()

    def test_stop_releases_device_on_capture_thread(self):
        cap, live = self.make_capture()
        live.release()
        self.assertTrue(cap.released.is_set())
        self.assertEqual(cap.release_thread, 'camera-capture')


class DemoLoopTests(unittest.TestCase):
    def run_demo(self, live_input):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        capture = Mock()
        capture.get.return_value = 30
        capture.read.return_value = (True, image)
        model = Mock()
        model.names = {0: 'person'}
        model.track.return_value = [SimpleNamespace(boxes=None)]
        live = Mock()
        sequence = iter(range(1, 20))
        live.latest.side_effect = lambda *args: SimpleNamespace(
            sequence=next(sequence), image=image, captured_at=time.monotonic())
        source = '0' if live_input else 'data/raw/crosswalk.mp4'
        roi_arg = 'config/__nonexistent_test_roi__.json' if live_input else 'config/crosswalk_roi.json'
        with patch('sys.argv', ['demo_camera.py', '--source', source, '--roi-file', roi_arg]), \
             patch.object(demo_camera, 'YOLO', return_value=model), \
             patch.object(demo_camera, 'open_video_capture', return_value=(capture, image)), \
             patch.object(demo_camera, 'LatestFrameCapture', return_value=live), \
             patch.object(demo_camera, 'select_roi', side_effect=AssertionError('unexpected static ROI prompt')), \
             patch.object(cv2, 'namedWindow'), patch.object(cv2, 'imshow') as show, \
             patch.object(cv2, 'getWindowProperty', return_value=1), \
             patch.object(cv2, 'waitKey', side_effect=[-1, -1, ord('q')]), \
             patch.object(cv2, 'destroyAllWindows'):
            demo_camera.main()
        self.assertEqual(model.track.call_count, 3)
        self.assertEqual(show.call_count, 3)
        if live_input:
            live.release.assert_called_once()
            capture.read.assert_not_called()
        else:
            self.assertEqual(capture.read.call_count, 2)
            capture.release.assert_called_once()

    def test_webcam_analyzes_continuously_without_roi_prompt(self):
        self.run_demo(live_input=True)

    def test_video_file_still_processes_frames_in_order(self):
        self.run_demo(live_input=False)


if __name__ == '__main__':
    unittest.main()
