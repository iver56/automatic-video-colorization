import unittest

from tcvc.gif import make_gif
from tests import TEST_FRAMES_DIR


class TestGif(unittest.TestCase):
    def test_make_gif(self):
        gif_path = TEST_FRAMES_DIR / "movie.gif"

        try:
            gif_path.unlink()  # Delete the file if it exists
        except FileNotFoundError:
            pass

        make_gif(TEST_FRAMES_DIR, max_num_frames=2)
        self.assertTrue(gif_path.exists())
        self.assertGreater(gif_path.stat().st_size, 90000)

        # Clean up
        gif_path.unlink()
