"""카메라 루프가 분석과 UI 이벤트 처리를 계속 수행하는지 검증합니다."""
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from scripts import demo_camera


class FakeLiveCapture:
    def __init__(self, packets):
        self.packets = iter(packets)
        self.release = Mock()

    def latest(self, after_sequence=0):
        try:
            return next(self.packets)
        except StopIteration:
            raise AssertionError('UI 이벤트 처리 없이 캡처 루프가 반복됩니다.')


class CameraLoopTests(unittest.TestCase):
    def run_loop(self, packet_ids, keys, expected_analyses, auto_roi=None):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        packets = [None if seq is None else SimpleNamespace(
            sequence=seq, image=image, captured_at=time.monotonic()) for seq in packet_ids]
        live = FakeLiveCapture(packets)
        capture = Mock()
        capture.get.return_value = 30
        model = Mock()
        model.names = {0: 'bicycle', 1: 'humans'}
        model.track.return_value = [SimpleNamespace(boxes=None)]
        with tempfile.TemporaryDirectory() as folder:
            roi_file = Path(folder) / 'camera.json'
            argv = ['demo_camera.py', '--source', '0', '--roi-file', str(roi_file)]
            with patch('sys.argv', argv), \
                 patch.object(demo_camera, 'YOLO', return_value=model), \
                 patch.object(demo_camera, 'open_video_capture', return_value=(capture, image)), \
                 patch.object(demo_camera, 'LatestFrameCapture', return_value=live), \
                 patch.object(demo_camera, 'auto_detect_crosswalk_roi', return_value=auto_roi) as auto, \
                 patch.object(demo_camera, 'select_roi', side_effect=AssertionError('Unexpected static ROI prompt')), \
                 patch.object(cv2, 'namedWindow'), patch.object(cv2, 'imshow') as show, \
                 patch.object(cv2, 'waitKey', side_effect=keys) as poll, \
                 patch.object(cv2, 'getWindowProperty', return_value=1), \
                 patch.object(cv2, 'destroyAllWindows'):
                demo_camera.main()
            self.assertEqual(poll.call_count, len(keys))
            self.assertEqual(model.track.call_count, expected_analyses)
            self.assertGreaterEqual(show.call_count, expected_analyses)
            self.assertTrue(all(call.kwargs['classes'] == [1] for call in model.track.call_args_list))
            auto.assert_called_once()
            if auto_roi is not None:
                self.assertEqual(json.loads(roi_file.read_text(encoding='utf-8'))['roi'], auto_roi)
        live.release.assert_called_once()
        capture.read.assert_not_called()

    def test_analysis_and_key_polling_repeat_until_quit(self):
        self.run_loop([1, 2, 3], [-1, -1, ord('q')], 3)

    def test_waiting_for_new_frame_still_polls_events(self):
        self.run_loop([1, None, 2], [-1, -1, ord('q')], 2)

    def test_pause_and_resume_still_poll_events(self):
        self.run_loop([1, 2, 3, 4], [ord('p'), -1, ord('p'), ord('q')], 2)

    def test_automatic_roi_is_preserved_and_analysis_starts(self):
        self.run_loop([1, 2], [-1, ord('q')], 2, auto_roi=[.1, .2, .6, .5])


if __name__ == '__main__':
    unittest.main()
