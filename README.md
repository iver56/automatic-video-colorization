# Automatic Temporally Coherent Video Colorization

[![Build status](https://img.shields.io/circleci/project/github/iver56/automatic-video-colorization/master.svg)](https://circleci.com/gh/iver56/automatic-video-colorization) [![Code coverage](https://img.shields.io/codecov/c/github/iver56/automatic-video-colorization/master.svg)](https://codecov.io/gh/iver56/automatic-video-colorization)

## Setup

`conda env create`

## Apply video colorization to a folder of PNG frames

`python -m tcvc.apply --input-path /path/to/images/`

## Run tests

`pytest`

## Licence and attribution

Credits go to [Harry-Thasarathan/TCVC](https://github.com/Harry-Thasarathan/TCVC). Changes made to the original can be found in the [commit history](https://github.com/iver56/automatic-video-colorization/commits/master). See also [the licence](https://github.com/iver56/automatic-video-colorization/blob/master/LICENCE.md).

The original paper can be read at [https://arxiv.org/abs/1904.09527](https://arxiv.org/abs/1904.09527).
