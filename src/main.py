#!/usr/bin/env python3
#
# Assess and display entropy of images via different methods.
# For usage, see `./main.py --help`.


import argparse
import math

from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np

from log import log
import methods


def parseargs():
    global args

    parser = argparse.ArgumentParser(description="Compute and display image entropy.")

    parser.add_argument(
        "-g",
        "--no-grey-image",
        help=f"do not display greyscale image on plot.",
        action="store_true",
    )

    parser.add_argument(
        "-i",
        "--no-input-image",
        help=f"do not display input image on plot.",
        action="store_true",
    )

    parser.add_argument(
        "-k",
        "--kernel-size",
        help="kernel size to be used with regional methods. must be a positive odd integer except 1. (default: 11)",
        type=argtypekernelsize,
        default=11,
    )

    parser.add_argument(
        "-r",
        "--radius",
        help="disk radius to be used with regional methods. must be an integer greater than 2. (default: 10)",
        type=argtyperadius,
        default=10,
    )

    parser.add_argument(
        "-m",
        "--method",
        help=f"method to use. possible values: {', '.join(methods.strtofunc.keys())} (default: {methods.default})",
        type=argtypemethod,
        default=methods.default,
    )

    parser.add_argument(
        "-w",
        "--white-background",
        help=f"display plots on a white background.",
        action="store_true",
    )

    parser.add_argument(
        "files",
        help="paths to input image files",
        nargs="+",
    )

    args = parser.parse_args()


def argtypemethod(val):
    if val not in methods.strtofunc.keys():
        raise argparse.ArgumentTypeError(f"{val} is not a valid method.")
    return val


def argtypekernelsize(val):
    ival = int(val)
    if ival <= 1 or ival % 2 != 1:
        raise argparse.ArgumentTypeError(f"{ival} is not a valid kernel size.")
    return ival


def argtyperadius(val):
    ival = int(val)
    if ival <= 2:
        raise argparse.ArgumentTypeError(f"{ival} is not a radius.")
    return ival


def plotall(entropy, colourimg, greyimg, plots):
    if colourimg is None or greyimg is None or plots is None:
        return False

    log("preparing figure")

    imgoffset = -int(args.no_input_image) - int(args.no_grey_image)
    nimg = len(plots) + 2 + imgoffset
    if nimg == 1:
        nx = 1
        ny = 1
    else:
        nx = nimg // 2
        ny = math.ceil(nimg / (nimg // 2))

    if not args.no_input_image:
        plt.subplot(nx, ny, 1)
        if colourimg.shape == greyimg.shape and np.all(colourimg == greyimg):
            plt.imshow(colourimg, cmap=plt.cm.gray)
        else:
            plt.imshow(colourimg)
        plt.title(f"Input Image")

    if not args.no_grey_image:
        plt.subplot(nx, ny, 2 + imgoffset)
        plt.imshow(greyimg, cmap=plt.cm.gray)
        plt.title(f"Greyscale Image")

    for i, plot in enumerate(plots):
        img, title, flags = plot
        plt.subplot(nx, ny, i + 3 + imgoffset)
        if "forcecolour" in flags:
            plt.imshow(img, cmap=plt.cm.jet)
        else:
            plt.imshow(img, cmap=plt.cm.gray)
        if "hasbar" in flags:
            plt.colorbar()
        plt.title(title)

    plt.suptitle(f"{args.method}: {entropy if entropy is not None else 'NaN'}")

    plt.tight_layout()

    return True


def main():
    parseargs()

    if not args.white_background:
        plt.style.use("dark_background")

    hasfigure = False
    nfl = len(args.files) - 1
    for i, fl in enumerate(args.files):
        log(f"processing file: {fl}")

        inputimg = cv.imread(fl)
        colourimg = cv.cvtColor(inputimg, cv.COLOR_BGR2RGB)  # for plotting
        greyimg = cv.cvtColor(inputimg, cv.COLOR_BGR2GRAY)

        assert greyimg.dtype == np.uint8, "image channel depth must be 8 bits"

        # prevent over/underflows during computation
        colourimg = colourimg.astype(np.int64)
        greyimg = greyimg.astype(np.int64)

        log("processing image")

        plt.figure(i + 1)
        hasfigure |= plotall(*methods.strtofunc[args.method](args, colourimg, greyimg))

        if i != nfl:
            print()

    if hasfigure:
        plt.show()
    else:
        log("no figure to show")


if __name__ == "__main__":
    main()
