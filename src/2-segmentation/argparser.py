from argparse import ArgumentParser, ArgumentTypeError

from util import die


def getargs():
    if hasattr(getargs, "args"):
        return getargs.args

    parser = ArgumentParser(description="Find ROIs in images and write to file.")

    modelgroup = parser.add_mutually_exclusive_group(required=True)
    optionalgroup = parser.add_argument_group()

    modelgroup.add_argument(
        "-a",
        "--semantic-model-ade20k",
        help="tensorflow semantic segmentation Ade20k model to use.",
    )

    optionalgroup.add_argument(
        "-f",
        "--infer-speed",
        help="speed to use for instance segmentation. (default: average)",
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

    modelgroup.add_argument(
        "-p",
        "--semantic-model-pascalvoc",
        help="tensorflow semantic segmentation Pascalvoc model to use.",
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
        "-t",
        "--mu",
        help="mu value to be used in distribution thresholding. (default: 0.99)",
        type=argtypeunit,
        default=0.99,
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

    if (
        not getargs.args.semantic_model_ade20k
        and not getargs.args.semantic_model_pascalvoc
    ):
        die("model path cannot be empty")

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
