#!/usr/bin/env python3
#
#
# Assess and display entropy of images via different methods.
# For usage, see `./main.py --help`.
#
#
###################################################################################################
# method              | resources                                                                 #
# =============================================================================================== #
# 2d-regional-shannon | TO BE ADDED                                                               #
# ----------------------------------------------------------------------------------------------- #
# 2d-gradient         | https://arxiv.org/abs/1609.01117                                          #
# ----------------------------------------------------------------------------------------------- #
# 2d-delentropy       | https://arxiv.org/abs/1609.01117                                          #
#                     | https://github.com/Causticity/sipp                                        #
# ----------------------------------------------------------------------------------------------- #
# 2d-scikit           | https://scikit-image.org/docs/dev/auto_examples/filters/plot_entropy.html #
#                     | https://scikit-image.org/docs/dev/api/skimage.filters.rank.html           #
# ----------------------------------------------------------------------------------------------- #
# 1d-shannon          |                                                                           #
# ----------------------------------------------------------------------------------------------- #
# 1d-kapur            | https://doi.org/10.1080/09720502.2020.1731976                             #
###################################################################################################


from copy import deepcopy as duplicate
from matplotlib import pyplot as plt
from skimage.filters.rank import entropy as skentropy
from skimage.morphology import disk as skdisk
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

    parser.add_argument(
        "-r",
        "--radius",
        help="disk radius to be used with regional methods. must be a positive integer except greater than 2. (default: 10)",
        type=argtyperadius,
        default=10,
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


def argtyperadius(val):
    ival = int(val)
    if ival <= 2:
        raise argparse.ArgumentTypeError(f"{ival} is not a radius.")
    return ival


def log(msg):
    print(f"{execname}: {msg}")


def plotall(colourimg, greyimg, plots):
    if colourimg is None or greyimg is None or plots is None:
        return False

    log("preparing figure")

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

    return True


def method_2d_regional_shannon(colourimg, greyimg):
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

    return (colourimg, greyimg, [(gradimg, "Gradient", [])])


def method_2d_delentropy(colourimg, greyimg):
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

    # TODO: is this how fx and fy are combined?
    #       it's for plotting and not used in computation anyways,
    #       and it matches the image in the article.
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

    ### 1609.01117 page 22

    deldensity = hist / np.sum(hist)
    deldensity = deldensity * -np.ma.log2(deldensity)
    entropy = np.sum(deldensity)
    entropy /= 2  # 4.3 Papoulis generalized sampling halves the delentropy

    # TODO: entropy is different from `sipp` and the article, but very similar
    log(f"entropy: {entropy}")
    log(f"entropy ratio: {entropy / 8.0}")

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
        ],
    )


def method_2d_regional_scikit(colourimg, greyimg):
    # From scikit docs:
    # The entropy is computed using base 2 logarithm i.e. the filter returns the minimum number of bits needed to encode the local gray level distribution.
    entimg = skentropy(greyimg, skdisk(args.radius))
    entropy = entimg.mean()

    log(f"entropy: {entropy}")
    log(f"entropy ratio: {entropy / 8.0}")

    return (colourimg, greyimg, [(entimg, "Scikit Entropy", ["hasbar"])])


def method_1d_shannon(colourimg, greyimg):
    signal = greyimg / greyimg.sum()
    entimg = signal * -np.ma.log2(signal)
    entropy = entimg.sum()

    log(f"entropy: {entropy}")
    log(f"entropy ratio: {entropy / 8.0}")

    return (colourimg, greyimg, [(entimg, "Shannon Entropy", [])])


def method_1d_kapur(colourimg, greyimg):
    hist = np.histogram(greyimg, bins=256, range=(0, 256))[0]
    cdf = hist.astype(float).cumsum()  # cumulative distribution function
    binrng = np.nonzero(hist)[0][[0, -1]]

    entropymax, threshold = 0, 0
    for i in range(binrng[0], binrng[1] + 1):
        histrng = hist[: i + 1] / cdf[i]
        entropy = -np.sum(histrng * np.ma.log(histrng))

        histrng = hist[i + 1 :]
        histrng = histrng[np.nonzero(histrng)] / (cdf[binrng[1]] - cdf[i])
        entropy -= np.sum(histrng * np.log(histrng))

        if entropy > entropymax:
            entropymax, threshold = entropy, i

    log(f"entropy: {entropy}")
    log(f"threshold: {threshold}")
    log(f"entropy ratio: {entropy / 8.0}")

    entimg = np.where(greyimg < threshold, greyimg, 0)

    return (colourimg, greyimg, [(entimg, "Kapur Threshold", [])])


def main():
    global execname
    global strtofunc

    execname = argv[0]

    strtofunc = {
        "2d-delentropy": method_2d_delentropy,
        "2d-regional-shannon": method_2d_regional_shannon,
        "2d-gradient": method_2d_regional_shannon,
        "2d-regional-scikit": method_2d_regional_scikit,
        "1d-shannon": method_1d_shannon,
        "1d-kapur": method_1d_kapur,
    }

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

        if greyimg.dtype != np.uint8:
            log("image must be one of 8-bit greyscale or 24/32-bit colour")
            exit(1)

        log("processing image")
        plt.figure(i + 1)
        hasfigure = hasfigure or plotall(*strtofunc[args.method](colourimg, greyimg))

        if nfl != i:
            print()

    if hasfigure:
        plt.show()
    else:
        log("no figure to show")


if __name__ == "__main__":
    main()
