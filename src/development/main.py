#!/usr/bin/env python3
# See `./main.py --help`.


import argparse
import multiprocessing as mp
import os
import sys
import time

from scipy.ndimage.filters import gaussian_filter
import cv2 as cv
import numpy as np
import scipy.stats as stats


def printerr(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def parseargs():
    global args

    parser = argparse.ArgumentParser(
        description="Find ROIs in images and write to file."
    )

    opgroup = parser.add_mutually_exclusive_group()

    parser.add_argument(
        "-k",
        "--kernel-size",
        help="kernel size to be used with regional methods. must be a positive odd integer except 1. (default: 11)",
        type=argtypekernelsize,
        default=11,
    )

    parser.add_argument(
        "-m",
        "--mu",
        help="mu value to be used in distribution thresholding. (default: 0.99)",
        type=argtypeunit,
        default=0.99,
    )

    parser.add_argument(
        "-s",
        "--sigma",
        help="sigma value for blurring the regions of interest. (default: 5.0)",
        type=argtypeposfloat,
        default=5,
    )

    parser.add_argument(
        "files",
        help="paths to input image files",
        metavar="FILE",
        nargs="+",
    )

    args = parser.parse_args()


def argtypekernelsize(val):
    ival = int(val)
    if ival <= 1 or ival % 2 != 1:
        raise argparse.ArgumentTypeError(f"{ival} is not a valid kernel size.")
    return ival


def argtypeunit(val):
    uval = float(val)
    if not (0.0 < uval < 1.0):
        raise argparse.ArgumentTypeError(f"{uval} is not a valid mu value.")
    return uval


def argtypeposfloat(val):
    fval = float(val)
    if fval <= 0:
        raise argparse.ArgumentTypeError(f"{fval} must be a positive real.")
    return fval


def processmain(fl):
    ### read

    inputimg = cv.imread(fl)

    if inputimg is None:  # do not use `not inputimg` for compatibility with arrays
        printerr(f"Could not read file {fl}.")
        return

    colourimg = cv.cvtColor(inputimg, cv.COLOR_BGR2RGB)  # for plotting
    greyimg = cv.cvtColor(inputimg, cv.COLOR_BGR2GRAY)

    assert greyimg.dtype == np.uint8, "image channel depth must be 8 bits"

    # prevent over/underflows during computation
    colourimg = colourimg.astype(np.int64)
    greyimg = greyimg.astype(np.int64)

    ### compute

    grad = np.gradient(greyimg)
    fx = grad[0].astype(int)
    fy = grad[1].astype(int)

    grad = fx + fy

    jrng = np.max([np.max(np.abs(fx)), np.max(np.abs(fy))])
    assert jrng <= 255, "J must be in range [-255, 255]"

    kernshape = (args.kernel_size,) * 2
    kerngrad = np.einsum(
        "ijkl->ij",
        np.lib.stride_tricks.as_strided(
            grad,
            tuple(np.subtract(grad.shape, kernshape) + 1) + kernshape,
            grad.strides * 2,
        ),
    )

    roigrad = np.abs(grad)
    roigradflat = roigrad.flatten()
    mean = np.mean(roigrad)
    roigradbound = (
        mean,
        *stats.t.interval(
            args.mu, len(roigradflat) - 1, loc=mean, scale=stats.sem(roigradflat)
        ),
    )
    roigrad[roigrad < roigradbound[2]] = 0
    roigrad[np.nonzero(roigrad)] = 255

    roigradblurred = gaussian_filter(roigrad, sigma=args.sigma)
    roigradblurred[np.nonzero(roigradblurred)] = 255

    roikerngrad = np.abs(grad)
    roikerngradflat = roikerngrad.flatten()
    mean = np.mean(roigrad)
    roikerngradbound = (
        mean,
        *stats.t.interval(
            args.mu,
            len(roikerngradflat) - 1,
            loc=mean,
            scale=stats.sem(roikerngradflat),
        ),
    )
    roikerngrad[roikerngrad < roikerngradbound[2]] = 0
    roikerngrad[np.nonzero(roikerngrad)] = 255

    roikerngradblurred = gaussian_filter(roikerngrad, sigma=args.sigma)
    roikerngradblurred[np.nonzero(roikerngradblurred)] = 255

    results = (
        (grad, "gradient"),
        (kerngrad, "gradient-convolved"),
        (roigrad, "roi"),
        (roigradblurred, "roi-blurred"),
        (roikerngrad, "roi-convolved"),
        (roikerngradblurred, "roi-convolved-blurred"),
    )

    ### write

    pathdir = f"{fl}_results"

    try:
        os.mkdir(pathdir)
    except FileExistsError:
        pass

    for r in results:
        data, name = r
        path = f"{pathdir}/{name}.png"
        if os.path.exists(path):
            os.remove(path)
        cv.imwrite(path, data)

    printerr(f"Done processing file {fl}.")


def main():
    parseargs()

    cpucount = os.cpu_count()
    processcount = int(cpucount * 13 / 12)

    print(
        "\n\t".join(
            [
                "Using parameters:",
                f"maximum process count: {processcount}",
                f"kernel size: {args.kernel_size}",
                f"sigma: {args.sigma}",
                f"mu: {args.mu}",
            ]
        )
    )

    timepassed = time.time()
    cputimepassed = time.process_time()

    with mp.Pool(processcount) as p:
        p.map(processmain, args.files)

    cputimepassed = time.process_time() - cputimepassed
    timepassed = time.time() - timepassed

    print(
        f"Processed {len(args.files)} files in {cputimepassed:.9g}s "
        + f"main process time and {timepassed:.9g}s real time."
    )


if __name__ == "__main__":
    main()
