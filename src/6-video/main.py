#!/usr/bin/env python3
#
# Assess and display entropy of images via different methods.
# For usage, see `./main.py --help`.


import argparse

from skvideo.io import vread
import numpy as np

import delentropy
import log


def parseargs():
    global args

    parser = argparse.ArgumentParser(description="Compute and display image entropy.")

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
        "-q",
        "--quiet",
        help="do not use stdout or stdin, unless -R is given or a fatal error is faced",
        action="store_true",
    )

    parser.add_argument(
        "files",
        help="paths to input video files",
        metavar="FILE",
        nargs="*",
    )

    args = parser.parse_args()

    reqlst = []
    if len(args.files) == 0:
        reqlst.append("FILE")
    if len(reqlst) != 0:
        parser.error(f"the following arguments are required: {', '.join(reqlst)}")


# TODO: rework to read frame-by-frame and keep a stack of n elements
# TODO: dynamically adjust size of stack
def processvideo(filename):
    log.info(f"reading {filename}")
    video = vread(filename)  # TODO: prone to running out of memory
    # video.shape -> (frames, height, width, 3) # TODO: sometimes width is up to +2 pixels off, why?
    log.info(f"video.shape = {video.shape}")

    log.info("assessing entropy")
    # TODO: implement joint entropy. converting to greyscale until implemented is acceptable.
    # TODO: delentropy functions' arguments and return values need work
    # TODO: delentropy functions might need some tidying for video
    delentropy.variation(args, video, video)  # TODO: prone to running out of memory


def main():
    parseargs()

    log.spongeout = args.sponge_out
    log.spongeerr = args.sponge_err
    if args.quiet:
        log.infoenabled = False
        log.warnenabled = False
        log.errenabled = False

    for file in args.files:
        processvideo(file)

    log.dumpcaches()


if __name__ == "__main__":
    main()
