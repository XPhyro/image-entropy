[![CodeFactor](https://www.codefactor.io/repository/github/xphyro/image-entropy/badge)](https://www.codefactor.io/repository/github/xphyro/image-entropy)

# image-entropy
This repository contains the source code of an ongoing research project on image
entropy assessment. More details will be given at a later stage of the research.

Under `src/1-assessment/`, the source code of the first phase of this research is
contained; namely, comparing and developing different image entropy assessment
methods and testing & benchmarking. The code requires Python >=3.6.

Under `src/2-segmentation/`, the chosen entropy assessment method is integrated
into a convolutional neural network. The code requires Python 3.7.

Under `src/3-recreation/`, a decoder for `src/development/`'s output is built. The
code requires Python >=3.6.

Under `src/4-cnn/`, CNN model tests are conducted. The code requires Python
\>=3.10.

All code are tested on Linux with an NVIDIA GPU. They may not work on other
systems.

# License
Unless otherwise specified, all code in this repository are licensed under
the MIT license. See the [LICENSE file](LICENSE) for details.
