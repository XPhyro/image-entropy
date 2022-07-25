#!/usr/bin/env python3
#
# Assess and display entropy of images via different methods.
# For usage, see `./main.py --help`.


import argparse
import math
import sys

from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np

import benchmark
import log
import methods
import testimages


def parseargs():
    global args

    parser = argparse.ArgumentParser(description="Compute and display image entropy.")

    opgroup = parser.add_mutually_exclusive_group()

    parser.add_argument(
        "-g",
        "--no-grey-image",
        help="do not display greyscale image on plot.",
        action="store_true",
    )

    parser.add_argument(
        "-H",
        "--test-height",
        help="test image height",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "-i",
        "--no-input-image",
        help="do not display input image on plot.",
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
        "-l",
        "--latex",
        help="output latex table rows for performance benchmarks",
        action="store_true",
    )

    parser.add_argument(
        "-m",
        "--method",
        help="method to use.",
        choices=list(methods.strtofunc.keys()),
    )

    opgroup.add_argument(
        "-n",
        "--noop",
        help="do not show or save plot.",
        action="store_true",
    )

    parser.add_argument(
        "-P",
        "--performance-count",
        help="number of iterations to use in performance metrics.",
        type=argtypeuint,
        default=1,
    )

    parser.add_argument(
        "-p",
        "--print-performance",
        help="print performance metrics.",
        action="store_true",
    )

    parser.add_argument(
        "-R",
        "--test-reshape",
        help="execute all tests with reshaping",
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
        help="save all test images to the given directory",
        type=str,
    )

    opgroup.add_argument(
        "-s",
        "--save",
        help="save the plots instead of showing",
        action="store_true",
    )

    parser.add_argument(
        "--sponge-out",
        help="batch print to stdout at the end",
        action="store_true",
    )

    parser.add_argument(
        "--sponge-err",
        help="batch print to stderr at the end",
        action="store_true",
    )

    parser.add_argument(
        "-t",
        "--use-tests",
        help=f"use generated test images instead of reading images from disk. possible test images are: {', '.join(testimages.strtofunc.keys())}",
        action="store_true",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        help="do not use stdout or stdin, unless -R is given or a fatal error is faced",
        action="store_true",
    )

    parser.add_argument(
        "-W",
        "--test-width",
        help="test image width",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "-w",
        "--white-background",
        help="display plots on a white background.",
        action="store_true",
    )

    parser.add_argument(
        "files",
        help="paths to input image files",
        metavar="FILE",
        nargs="*",
    )

    args = parser.parse_args()

    if args.save_tests is None and not args.test_reshape:
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
        if nx > ny:
            nx, ny = ny, nx

    if not args.no_input_image:
        plt.subplot(nx, ny, 1)
        if colourimg.shape == greyimg.shape and np.all(colourimg == greyimg):
            plt.imshow(colourimg, cmap=plt.cm.gray)
        else:
            plt.imshow(colourimg)
        plt.title("Input Image")

    if not args.no_grey_image:
        plt.subplot(nx, ny, 2 + imgoffset)
        plt.imshow(greyimg, cmap=plt.cm.gray)
        plt.title("Greyscale Image")

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
    methods.init(args)

    log.spongeout = args.sponge_out
    log.spongeerr = args.sponge_err
    if args.quiet:
        log.infoenabled = False
        log.warnenabled = False
        log.errenabled = False

    if not args.white_background:
        plt.style.use("dark_background")

    if args.test_reshape:
        entropyarrs = []
        widtharrs = []
        heightarrs = []

        for i, (testname, testfunc) in enumerate(testimages.strtofunc.items()):
            plt.figure(i + 1)
            entropies = []
            widths = []
            heights = []

            width = args.test_width
            height = args.test_height
            while width >= 2:
                greyimg = testfunc(width, height)
                colourimg = cv.cvtColor(greyimg, cv.COLOR_GRAY2RGB)
                greyimg = greyimg.astype(np.int64)
                colourimg = colourimg.astype(np.int64)

                entropy = methods.strtofunc[args.method](args, colourimg, greyimg)[0]

                entropies.append(entropy)
                widths.append(width)
                heights.append(height)

                width = int(width / 2)
                height = int(height * 2)

            plt.plot(np.arange(len(widths)), entropies)
            plt.title(f"{args.method}: {testname}")
            plt.tight_layout()
            plt.savefig(f"{args.method}_{testname}.pdf", bbox_inches="tight")

            entropyarrs.append(entropies)
            widtharrs.append(widths)
            heightarrs.append(heights)

        print(f"entarrs = {repr(entropyarrs)}")
        print(f"warrs = {repr(widtharrs)}")
        print(f"harrs = {repr(heightarrs)}")

        sys.exit(0)

    if args.save_tests is not None:
        for s, f in testimages.strtofunc.items():
            cv.imwrite(f"{args.save_tests}/{s}.png", f())
        sys.exit(0)

    log.info(f"selected method: {args.method}")

    hasfigure = False
    nfl = len(args.files) - 1
    for i, fl in enumerate(args.files):
        log.info(f"processing file: {fl}")

        if args.use_tests:
            greyimg = testimages.strtofunc[fl](args.test_width, args.test_height)
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
            log._print("", end="")

    if not args.noop and not args.save:
        if hasfigure:
            log.dumpcaches()
            plt.show()
        else:
            log.info("no figure to show")
            log.dumpcaches()
    else:
        log.dumpcaches()


if __name__ == "__main__":
    main()
