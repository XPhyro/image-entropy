#!/usr/bin/env python3
#
# Assess and display entropy of images via different methods.
# For usage, see `./main.py --help`.


import argparse
import math

from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np

import log
import benchmark
import methods
import testimages


def parseargs():
    global args

    parser = argparse.ArgumentParser(description="Compute and display image entropy.")

    opgroup = parser.add_mutually_exclusive_group()

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
        "-m",
        "--method",
        help=f"method to use.",
        choices=list(methods.strtofunc.keys()),
    )

    opgroup.add_argument(
        "-n",
        "--noop",
        help=f"do not show or save plot.",
        action="store_true",
    )

    parser.add_argument(
        "-P",
        "--performance-count",
        help=f"number of iterations to use in performance metrics.",
        type=argtypeuint,
        default=1,
    )

    parser.add_argument(
        "-p",
        "--print-performance",
        help=f"print performance metrics.",
        action="store_true",
    )

    parser.add_argument(
        "-r",
        "--radius",
        help="disk radius to be used with regional methods. must be an integer greater than 2. (default: 10)",
        type=argtyperadius,
        default=10,
    )

    parser.add_argument(
        "-S",
        "--save-tests",
        help=f"save all test images to the given directory",
        type=str,
    )

    opgroup.add_argument(
        "-s",
        "--save",
        help=f"save the plots instead of showing",
        action="store_true",
    )

    parser.add_argument(
        "-t",
        "--use-tests",
        help=f"use generated test images instead of reading images from disk. possible test images are: {', '.join(testimages.strtofunc.keys())}",
        action="store_true",
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
        metavar="FILE",
        nargs="*",
    )

    args = parser.parse_args()

    if args.save_tests is None:
        reqlst = []
        if args.method is None:
            reqlst.append("-m/--method")
        if len(args.files) == 0:
            reqlst.append("FILE")
        if len(reqlst) != 0:
            parser.error(f"the following arguments are required: {', '.join(reqlst)}")


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


def argtypeuint(val):
    ival = int(val)
    if ival < 0:
        raise argparse.ArgumentTypeError(f"{ival} is not a non-negative integer.")
    return ival


def plotall(entropy, colourimg, greyimg, plots):
    if colourimg is None or greyimg is None or plots is None:
        return False

    log.info("preparing figure")

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

    plt.suptitle(f"{args.method}\n$H$ = {entropy if entropy is not None else 'NaN'}")

    plt.tight_layout()

    return True


def main():
    parseargs()

    if args.save_tests is not None:
        for s, f in testimages.strtofunc.items():
            cv.imwrite(f"{args.save_tests}/{s}.png", f())
        exit(0)

    if not args.white_background:
        plt.style.use("dark_background")

    hasfigure = False
    nfl = len(args.files) - 1
    for i, fl in enumerate(args.files):
        log.info(f"processing file: {fl}")

        if args.use_tests:
            greyimg = testimages.strtofunc[fl]()
            colourimg = cv.cvtColor(greyimg, cv.COLOR_GRAY2RGB)  # for plotting
        else:
            inputimg = cv.imread(fl)
            colourimg = cv.cvtColor(inputimg, cv.COLOR_BGR2RGB)  # for plotting
            greyimg = cv.cvtColor(inputimg, cv.COLOR_BGR2GRAY)

            assert greyimg.dtype == np.uint8, "image channel depth must be 8 bits"

        # prevent over/underflows during computation
        colourimg = colourimg.astype(np.int64)
        greyimg = greyimg.astype(np.int64)

        log.info("processing image")

        plt.figure(i + 1)
        hasfigure |= plotall(*methods.strtofunc[args.method](args, colourimg, greyimg))

        log.info("benchmarking performance")
        benchmark.benchmark(
            args, methods.strtofunc[args.method], (args, colourimg, greyimg)
        )

        if args.save:
            plt.savefig(f"{args.method}_{fl}.pdf", bbox_inches="tight")

        if i != nfl:
            print()

    if not args.noop and not args.save:
        if hasfigure:
            plt.show()
        else:
            log.info("no figure to show")


if __name__ == "__main__":
    main()
