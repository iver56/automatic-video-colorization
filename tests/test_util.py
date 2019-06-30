import numpy as np
import unittest

from tcvc.util import load_img
from tests import TEST_FRAMES_DIR


class TestUtil(unittest.TestCase):
    def test_load_image(self):
        img = load_img(TEST_FRAMES_DIR / "subfolder" / "frame008.png")

        self.assertEqual(img.dtype, np.uint8)
        self.assertEqual(img.shape, (256, 256, 3))
        self.assertEqual(np.amin(img[:, :, 0]), 15)
        self.assertEqual(np.amax(img[:, :, 1]), 252)
