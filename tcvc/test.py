import unittest

from tcvc.dataset import DatasetFromFolder


class TestDatasetFromFolder(unittest.TestCase):
    def test_parse_frame_number(self):
        filename = "22orCRlH-TI_00001.png"
        frame_number, padded_frame_number = DatasetFromFolder.get_frame_number(filename)
        self.assertEqual(frame_number, 1)
        self.assertEqual(padded_frame_number, "00001")

        filename = "frame8567.jpg"
        frame_number, padded_frame_number = DatasetFromFolder.get_frame_number(filename)
        self.assertEqual(frame_number, 8567)
        self.assertEqual(padded_frame_number, "8567")

        filename = "0002336.png"
        frame_number, padded_frame_number = DatasetFromFolder.get_frame_number(filename)
        self.assertEqual(frame_number, 2336)
        self.assertEqual(padded_frame_number, "0002336")

    def test_previous_frame_filename(self):
        file_path = "/tmp/22orCRlH-TI_00444.png"
        previous_frame_file_path = DatasetFromFolder.get_previous_frame_file_path(
            file_path
        )
        self.assertEqual(previous_frame_file_path.name, "22orCRlH-TI_00443.png")

        file_path = "D:/code/demo-style/data/content_images/zeven-bw/zeven/frame004.png"
        previous_frame_file_path = DatasetFromFolder.get_previous_frame_file_path(
            file_path
        )
        self.assertEqual(previous_frame_file_path.name, "frame003.png")
