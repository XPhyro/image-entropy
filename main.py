#!/usr/bin/env python3
#
#
# Assess and display entropy of images via different methods.
# For usage, see `./main.py --help`.
#
#
#######################################################################
# method              | resources                                     #
# =================================================================== #
# 2d-regional-shannon | TO BE ADDED                                   #
# ------------------------------------------------------------------- #
# 2d-gradient         | https://arxiv.org/abs/1609.01117              #
# ------------------------------------------------------------------- #
# 2d-delentropy       | https://arxiv.org/abs/1609.01117              #
#                     | https://github.com/Causticity/sipp            #
# ------------------------------------------------------------------- #
# 1d-kapur            | https://doi.org/10.1080/09720502.2020.1731976 #
#######################################################################


from copy import deepcopy as duplicate
from matplotlib import pyplot as plt
from operator import itemgetter
from sys import argv
import argparse
import cv2 as cv
import math
import numpy as np


def parseargs():
    global args

    parser = argparse.ArgumentParser(description="Compute and display image entropy.")

    parser.add_argument(
        "-k",
        "--kernel-size",
        help="kernel size to be used with regional methods. must be a positive odd integer except 1. (default: 11)",
        type=argtypekernelsize,
        default=11,
    )

    defaultmethod = list(strtofunc.keys())[0]
    parser.add_argument(
        "-m",
        "--method",
        help=f"method to use. possible values: {', '.join(strtofunc.keys())} (default: {defaultmethod})",
        type=argtypemethod,
        default=defaultmethod,
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
    if val not in strtofunc.keys():
        raise argparse.ArgumentTypeError(f"{val} is not a valid method.")
    return val


def argtypekernelsize(val):
    ival = int(val)
    if ival <= 1 or ival % 2 != 1:
        raise argparse.ArgumentTypeError(f"{ival} is not a valid kernel size.")
    return ival


def log(msg):
    print(f"{execname}: {msg}")


def plotall(colourimg, greyimg, plots):
    nimg = len(plots) + 2
    nx = nimg // 2
    ny = math.ceil(nimg / (nimg // 2))

    plt.subplot(nx, ny, 1)
    if colourimg.shape == greyimg.shape and np.all(colourimg == greyimg):
        plt.imshow(colourimg, cmap=plt.cm.gray)
    else:
        plt.imshow(colourimg)
    plt.title(f"Input Image")

    plt.subplot(nx, ny, 2)
    plt.imshow(greyimg, cmap=plt.cm.gray)
    plt.title(f"Greyscale Image")

    for i, plot in enumerate(plots):
        img, title, flags = plot
        plt.subplot(nx, ny, i + 3)
        if "forcecolour" in flags:
            plt.imshow(img, cmap=plt.cm.jet)
        else:
            plt.imshow(img, cmap=plt.cm.gray)
        if "hasbar" in flags:
            plt.colorbar()
        plt.title(title)


def method_2d_regional_shannon(colourimg, greyimg):
    log("processing image")

    entimg = duplicate(greyimg)
    imgshape = entimg.shape

    kernsize = args.kernel_size
    kernrad = round((kernsize - 1) / 2)

    entropies = []
    for i in range(imgshape[0]):
        for j in range(imgshape[1]):
            region = greyimg[
                # ymax:ymin, xmax:xmin
                np.max([0, i - kernrad]) : np.min([imgshape[0], i + kernrad]),
                np.max([0, j - kernrad]) : np.min([imgshape[1], j + kernrad]),
            ].flatten()
            size = region.size

            probs = [np.size(region[region == i]) / size for i in set(region)]
            entropy = np.sum([p * np.log2(1 / p) for p in probs])

            entropies.append(entropy)
            entimg[i, j] = entropy

    log(f"entropy = {np.average(entropies)} ± {np.std(entropies)}")

    log("preparing figure")

    return (
        colourimg,
        greyimg,
        [
            (
                entimg,
                "Entropy Map With {kernsize}x{kernsize} Kernel",
                ["hasbar", "forcecolour"],
            )
        ],
    )


def method_2d_gradient(colourimg, greyimg):
    log("processing image")

    param_realgrad = True
    param_concave = True

    if param_realgrad:
        grads = np.gradient(greyimg)
        gradx = grads[0]
        grady = grads[1]
    else:
        gradx = cv.filter2D(
            greyimg,
            cv.CV_8U,
            cv.flip(np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]), -1),
            borderType=cv.BORDER_CONSTANT,
        )
        grady = cv.filter2D(
            greyimg,
            cv.CV_8U,
            cv.flip(np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]), -1),
            borderType=cv.BORDER_CONSTANT,
        )

    gradimg = (
        np.bitwise_or(gradx, grady)
        if not param_realgrad
        else (
            gradx + grady
            if not param_concave
            else np.invert(np.array(gradx + grady, dtype=int))
        )
    )

    log(f"gradient = {np.average(gradimg)} ± {np.std(gradimg)}")

    log("preparing figure")

    return (colourimg, greyimg, [(gradimg, "Gradient", [])])


def method_2d_delentropy(colourimg, greyimg):
    log("processing image")

    ### 1609.01117 page 10

    # if set to True, use method explained in the article
    # else, use alternative method
    param_diffgrad = True

    if param_diffgrad:
        # $\nabla f(n) \approx f(n) - f(n - 1)$
        fx = greyimg[:, 2:] - greyimg[:, :-2]
        fy = greyimg[2:, :] - greyimg[:-2, :]
        # fix shape
        fx = fx[1:-1, :]
        fy = fy[:, 1:-1]
    else:
        grad = np.gradient(greyimg)
        fx = grad[0].astype(int)
        fy = grad[1].astype(int)

    # TODO: is this how fx and fy are combined? (it's for plotting and not used in computation anyways)
    grad = fx + fy

    # ensure $-255 \leq J \leq 255$
    jrng = np.max([np.max(np.abs(fx)), np.max(np.abs(fy))])
    assert jrng <= 255

    ### 1609.01117 page 16

    hist, edgex, edgey = np.histogram2d(
        fx.flatten(),
        fy.flatten(),
        bins=2 * jrng + 1,
        range=[[-jrng, jrng], [-jrng, jrng]],
    )

    deldensity = hist / np.sum(hist)
    entimg = deldensity * -np.ma.log2(deldensity)
    entimg /= 2  # 4.3 Papoulis generalized sampling halves the delentropy
    entropy = np.sum(entimg)

    # TODO: entropy is different from `sipp` and the article, but very similar
    log(f"entropy: {entropy}")
    log(f"entropy ratio: {entropy / 8.0}")

    log("preparing figure")

    # the reference image seems to be bitwise inverted, I don't know why.
    # the entropy doesn't change when inverted, so both are okay in
    # the previous computational steps.
    param_invert = True

    gradimg = np.invert(grad) if param_invert else grad

    return (
        colourimg,
        greyimg,
        [
            (gradimg, "Gradient", []),
            (deldensity, "Deldensity", ["hasbar", "forcecolour"]),
            (entimg, "Delentropy", ["hasbar", "forcecolour"]),
        ],
    )


def method_1d_kapur(colourimg, greyimg):
    log("processing image")

    hist = np.histogram(greyimg, bins=256, range=(0, 256))[0]
    cdf = hist.astype(float).cumsum()
    ibin, fbin = itemgetter(0, -1)(np.nonzero(hist)[0])

    entropymax, threshold = 0, 0
    for i in range(ibin, fbin + 1):
        histrng = hist[: i + 1] / cdf[i]
        entropy = -np.sum(histrng * np.ma.log(histrng))

        histrng = hist[i + 1 :]
        histrng = histrng[np.nonzero(histrng)] / (cdf[fbin] - cdf[i])
        entropy -= np.sum(histrng * np.log(histrng))

        if entropy > entropymax:
            entropymax, threshold = entropy, i

    log(f"entropy: {entropy}")
    log(f"entropy ratio: {entropy / 8.0}")


def main():
    global execname
    global strtofunc

    execname = argv[0]

    strtofunc = {
        "2d-delentropy": method_2d_delentropy,
        "2d-regional-shannon": method_2d_regional_shannon,
        "2d-gradient": method_2d_regional_shannon,
        "1d-kapur": method_1d_kapur,
    }

    parseargs()

    if not args.white_background:
        plt.style.use("dark_background")

    nfl = len(args.files) - 1
    for i, fl in enumerate(args.files):
        log(f"processing file: {fl}")

        inputimg = cv.imread(fl)
        colourimg = cv.cvtColor(inputimg, cv.COLOR_BGR2RGB).astype(int)  # for plotting
        greyimg = cv.cvtColor(inputimg, cv.COLOR_BGR2GRAY).astype(int)

        plt.figure(i + 1)
        plotall(*strtofunc[args.method](colourimg, greyimg))

        if nfl != i:
            print()

    plt.show()


if __name__ == "__main__":
    main()
