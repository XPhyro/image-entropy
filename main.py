#!/usr/bin/env python3
#
# Assess and display entropy of images via different methods.
# For usage, see `./main.py --help`.
#
# method           resources
#
# pseudo-spatial   https://stats.stackexchange.com/questions/235270/entropy-of-an-image
#                  https://stackoverflow.com/questions/50313114/what-is-the-entropy-of-an-image-and-how-is-it-calculated


import argparse
from copy import deepcopy as duplicate
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def main():
    inputimg = Image.open(args.input)
    greyimg = inputimg.convert("L")
    inputimgarr = np.array(inputimg)
    greyimgarr = np.array(greyimg)

    imgarr = duplicate(greyimgarr)
    imgshape = imgarr.shape

    kernsize = args.kernel_size
    kerndist = round((kernsize - 1) / 2)

    entropies = []

    if args.method == "pseudo-spatial":
        for i in range(imgshape[0]):
            for j in range(imgshape[1]):
                region = greyimgarr[
                    # ymax:ymin, xmax:xmin
                    np.max([0, i - kerndist]) : np.min([imgshape[0], i + kerndist]),
                    np.max([0, j - kerndist]) : np.min([imgshape[1], j + kerndist]),
                ].flatten()
                size = region.size

                probs = [np.size(region[region == i]) / size for i in set(region)]
                entropy = np.sum([p * np.log2(1 / p) for p in probs])

                entropies.append(entropy)
                imgarr[i, j] = entropy

    print(f"Entropy: {np.average(entropies)} ± {np.std(entropies)}")

    plt.subplot(1, 3, 1)
    plt.imshow(inputimgarr)
    plt.title(f"Input Image")

    plt.subplot(1, 3, 2)
    plt.imshow(greyimgarr, cmap=plt.cm.gray)
    plt.title(f"Greyscale Image")

    plt.subplot(1, 3, 3)
    plt.imshow(imgarr, cmap=plt.cm.jet)
    plt.colorbar()
    plt.title(f"Entropy Map With {kernsize}x{kernsize} Kernel")

    plt.show()


def argtypemethod(val):
    if val != "pseudo-spatial":
        raise argparse.ArgumentTypeError(f"{val} is not a valid method.")
    return val


def argtypekernelsize(val):
    ival = int(val)
    if ival <= 1 or ival % 2 != 1:
        raise argparse.ArgumentTypeError(f"{ival} is not a valid kernel size.")
    return ival


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Entropy")

    parser.add_argument("-i", "--input", help="path to input image file", type=str)
    parser.add_argument(
        "-k",
        "--kernel-size",
        help="kernel size. must be a positive odd integer. (default: 11, minimum: 3)",
        type=argtypekernelsize,
        default=11,
    )
    parser.add_argument(
        "-m",
        "--method",
        help="method to use. possible values: pseudo-spatial (default: pseudo-spatial)",
        type=argtypemethod,
        default="pseudo-spatial",
    )
    # parser.add_argument("files", help="paths to input image files", nargs="*")

    args = parser.parse_args()

    main()
