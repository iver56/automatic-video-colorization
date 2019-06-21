import argparse
import os
import subprocess
from pathlib import Path

from tcvc.util import get_image_file_paths

TMP_DIR = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "tmp"
)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Convert a set of frames to a video file (MP4 with x264 encoding)"
    )
    arg_parser.add_argument(
        "--input-path",
        dest="input_path",
        help="Path to the folder where the images frames) reside",
        type=str,
        required=True,
    )
    arg_parser.add_argument(
        "--framerate",
        dest="framerate",
        help="Specify the framerate as a positive integer",
        type=int,
        required=False,
        default=30,
    )

    args = arg_parser.parse_args()

    assert args.framerate > 1

    image_file_paths = get_image_file_paths(args.input_path)
    image_list_file_path = os.path.join(TMP_DIR, "images.txt")
    os.makedirs(TMP_DIR, exist_ok=True)
    with open(image_list_file_path, "w") as f:
        for path in image_file_paths:
            f.write("file {}\n".format(Path(path).as_posix()))

    subprocess.run(
        [
            "ffmpeg",
            "-r",
            "{}".format(args.framerate),
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            image_list_file_path,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "{}".format(Path(os.path.join(args.input_path, "video.mp4")).as_posix()),
        ]
    )
