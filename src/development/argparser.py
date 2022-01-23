from argparse import ArgumentParser, ArgumentTypeError


def getargs():
    if hasattr(getargs, "args"):
        return getargs.args

    parser = ArgumentParser(description="Find ROIs in images and write to file.")

    requiredgroup = parser.add_argument_group("required arguments")
    optionalgroup = parser.add_argument_group("optional arguments")

    optionalgroup.add_argument(
        "-f",
        "--infer-speed",
        help="speed to use. (default: average)",
        choices=["average", "fast", "rapid"],
        default="average",
    )

    optionalgroup.add_argument(
        "-k",
        "--kernel-size",
        help="kernel size to be used with regional methods. must be a positive odd integer except 1. (default: 11)",
        type=argtypekernelsize,
        default=11,
    )

    optionalgroup.add_argument(
        "-M",
        "--mu",
        help="mu value to be used in distribution thresholding. (default: 0.99)",
        type=argtypeunit,
        default=0.99,
    )

    requiredgroup.add_argument(
        "-m",
        "--model",
        help="tensorflow instance segmentation model to use.",
        required=True,
    )

    optionalgroup.add_argument(
        "-S",
        "--save-images",
        help="save debugging images.",
        action="store_true",
    )

    optionalgroup.add_argument(
        "-s",
        "--sigma",
        help="sigma value for blurring the regions of interest. (default: 5.0)",
        type=argtypeposfloat,
        default=5,
    )

    optionalgroup.add_argument(
        "-z",
        "--zero-terminated",
        help="file delimiter in standard input is NUL, not newline.",
        action="store_true",
    )

    optionalgroup.add_argument(
        "files",
        help="paths to input image files. with no FILE, read standard input.",
        metavar="FILE",
        nargs="*",
    )

    getargs.args = parser.parse_args()
    return getargs.args


def argtypekernelsize(val):
    ival = int(val)
    if ival <= 1 or ival % 2 != 1:
        raise ArgumentTypeError(f"{ival} is not a valid kernel size.")
    return ival


def argtypeunit(val):
    uval = float(val)
    if not 0.0 < uval < 1.0:
        raise ArgumentTypeError(f"{uval} is not a valid mu value.")
    return uval


def argtypeposfloat(val):
    fval = float(val)
    if fval <= 0:
        raise ArgumentTypeError(f"{fval} must be a positive real.")
    return fval
