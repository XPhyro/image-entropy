import argparse

import methods
import testimages


def getargs():
    if hasattr(getargs, "args"):
        return getargs.args

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

    getargs.args = parser.parse_args()

    if getargs.args.save_tests is None and not getargs.args.test_reshape:
        reqlst = []
        if getargs.args.method is None:
            reqlst.append("-m/--method")
        if len(getargs.args.files) == 0:
            reqlst.append("FILE")
        if len(reqlst) != 0:
            parser.error(f"the following arguments are required: {', '.join(reqlst)}")

    return getargs.args


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
