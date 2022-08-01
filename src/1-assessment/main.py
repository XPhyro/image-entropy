#!/usr/bin/env python3
#
# Assess and display entropy of images via different methods.
# For usage, see `./main.py --help`.


import math
import sys

from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np

import argparser
import benchmark
import log
import methods
import testimages


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
    global args
    args = argparser.getargs()

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
            plt.savefig(f"{testname}_{args.method}.pdf", bbox_inches="tight")

            entropyarrs.append(entropies)
            widtharrs.append(widths)
            heightarrs.append(heights)

        print(f"entarrs = {repr(entropyarrs)}")
        print(f"warrs = {repr(widtharrs)}")
        print(f"harrs = {repr(heightarrs)}")

        sys.exit(0)

    if args.save_tests is not None:
        for s, f in testimages.strtofunc.items():
            cv.imwrite(
                f"{args.save_tests}/{s}.png", f(args.test_width, args.test_height)
            )
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
