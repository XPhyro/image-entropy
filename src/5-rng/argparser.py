from argparse import ArgumentParser


def getargs():
    if hasattr(getargs, "args"):
        return getargs.args

    parser = ArgumentParser(
        description="Generate pseudo random numbers using ChaCha and SHA."
    )

    optionalgroup = parser.add_argument_group()

    optionalgroup.add_argument(
        "-c",
        "--count",
        help="count of random numbers to produce",
        type=int,
        default=10,
    )

    getargs.args = parser.parse_args()

    return getargs.args
