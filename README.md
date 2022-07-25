[![CodeFactor](https://www.codefactor.io/repository/github/xphyro/image-entropy/badge)](https://www.codefactor.io/repository/github/xphyro/image-entropy)

# image-entropy
This repository contains the source code of an ongoing research project on
image/video entropy assessment.

Under `src/1-assessment/`, the source code of the first phase of this research
is contained; namely, comparing and developing different image entropy
assessment methods and testing & benchmarking. Requires Python >=3.10.

Under `src/2-segmentation/`, the chosen entropy assessment method is integrated
into a convolutional neural network. Requires Python 3.7.

Under `src/3-recreation/`, a decoder for `src/development/`'s output is built.
Requires Python >=3.6.

Under `src/4-cnn/`, CNN model tests are conducted. Requires Python 3.10.

Under `src/5-random/`, random number generation tests are conducted. Requires
Python \>=3.6.

Under `src/6-video/`, `1-assessment/` and `2-segmentation` are applied to video.
Requires Python \>=3.6.

All code are tested on Linux with an NVIDIA GPU. They may not work on other
systems.

# License
Unless otherwise specified, all code in this repository are licensed under
the MIT license. See the [LICENSE file](LICENSE) for details.
