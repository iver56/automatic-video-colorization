# Automatic Temporally Coherent Video Colorization

[![Build status](https://img.shields.io/circleci/project/github/iver56/automatic-video-colorization/master.svg)](https://circleci.com/gh/iver56/automatic-video-colorization) [![Code coverage](https://img.shields.io/codecov/c/github/iver56/automatic-video-colorization/master.svg)](https://codecov.io/gh/iver56/automatic-video-colorization)

## Setup

`conda env create`

Also note that some of the scripts depend on `ffmpeg`.

## Train a model on a folder of PNG frames

`python -m tcvc.train --dataset-path /path/to/images/ --input-style line_art`

or

`python -m tcvc.train --dataset-path /path/to/images/ --input-style greyscale`


The frame filenames should have zero-padded frame numbers, for example like this:

* frame00001.png, frame00002.png, frame00003.png, ...

If you have multiple sequences of frames (i.e. from different videos/scenes/shots), you can have different prefixes in the frame filenames, like this:
* firstvideo00001.png, firstvideo00002.png, firstvideo00003.png, ..., secondvideo00001.png, secondvideo00002.png, secondvideo00003.png, ...

Alternatively, the different frame sequences can reside in different subfolders. For that to work, you have to use the `--include-subfolders` argument.

## Apply video colorization to a folder of PNG frames

`python -m tcvc.apply --input-path /path/to/images/ --input-style line_art`

By default, this command will use a model that is included in this repository. It is trained on Dragonball line art. If you want to specify a different model, you can do that with `--model`.

## Run tests

`pytest`

## Licence and attribution

Credits go to [Harry-Thasarathan/TCVC](https://github.com/Harry-Thasarathan/TCVC). Changes made to the original can be found in the [commit history](https://github.com/iver56/automatic-video-colorization/commits/master). See also [the licence](https://github.com/iver56/automatic-video-colorization/blob/master/LICENCE.md).

The original paper by Harrish Thasarathan, Kamyar Nazeri and Mehran Ebrahimi (2019) can be read at [https://arxiv.org/abs/1904.09527](https://arxiv.org/abs/1904.09527).

## Known issues

* The code is not compatible with CPU mode as of 2019-06-21
