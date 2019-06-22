import shutil
import unittest

from tcvc.extract_video_frames import extract_video_frames, parse_args
from tests import TEST_FRAMES_DIR


class TestExtractVideoFrames(unittest.TestCase):
    def test_extract_video_frames(self):
        try:
            shutil.rmtree(TEST_FRAMES_DIR / "video_frames")
        except FileNotFoundError:
            pass
        extract_video_frames(TEST_FRAMES_DIR / "video.mp4")
        self.assertTrue((TEST_FRAMES_DIR / "video_frames").is_dir())
        shutil.rmtree(TEST_FRAMES_DIR / "video_frames")

    def test_parse_args(self):
        args = parse_args(['--input-path', TEST_FRAMES_DIR.as_posix()])
        self.assertIn('frames', args.input_path)
