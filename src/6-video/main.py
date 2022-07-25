#!/usr/bin/env python3
#
# Assess and display entropy of images via different methods.
# For usage, see `./main.py --help`.


from operator import itemgetter
import argparse
import subprocess as sp

import ffmpeg as ffm
import numpy as np

import delentropy
import log


def parseargs():
    global args

    parser = argparse.ArgumentParser(description="Compute and display image entropy.")

    parser.add_argument(
        "-b",
        "--buffer-size",
        help="how many frames to buffer",
        type=argtypepint,
        default=4,
    )

    parser.add_argument(
        "-g",
        "--greyscale",
        help="convert frames to greyscale",
        action="store_true",
    )

    parser.add_argument(
        "-n",
        "--max-frame-count",
        help="limit number of frames processed per stream per file. 0 is unlimited.",
        type=argtypeuint,
        default=0,
    )

    parser.add_argument(
        "-S",
        "--strict-stack",
        help="do not process stack if stack size is not at effective maximum",
        action="store_true",
    )

    parser.add_argument(
        "-s",
        "--max-stack-size",
        help="maximum size of the frame stack in count of frames. 0 is unlimited. effective maximum size may be smaller than the given maximum size.",
        type=argtypeuint,
        default=8,
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
        "-q",
        "--quiet",
        help="do not use stdout and stderr",
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

    if not args.greyscale:
        log.warn(
            "joint entropy is not yet implemented, colour frame assessment will be inaccurate. "
            + "consider using --greyscale."
        )


def argtypeuint(val):
    ival = int(val)
    if ival < 0:
        raise argparse.ArgumentTypeError(f"argument must be a non-negative integer.")
    return ival


def argtypepint(val):
    ival = int(val)
    if ival <= 0:
        raise argparse.ArgumentTypeError(f"argument must be a positive integer.")
    return ival


def processvideo(filename):
    log.info(f"reading {filename}")
    streams = ffm.probe(filename, select_streams="v")["streams"]
    nstreams = len(streams)
    if nstreams == 0:
        log.err("no video streams found in file, skipping")
        return

    log.info(f"found {nstreams} {'stream' if nstreams == 1 else 'streams'}")
    for stream in streams:
        streamidx = stream["index"]
        streamhidx = streamidx + 1

        log.info(f"processing stream {streamhidx}/{nstreams}")

        shape = itemgetter("height", "width")(stream)
        if not args.greyscale:
            shape = (*shape, 3)
        shape = np.array(shape)
        framesize = int(np.prod(shape, dtype=int))
        log.info(f"shape: {shape}", f"frame size: {framesize}")

        argv = (
            ffm.input(filename)
            .output(
                "pipe:",
                map=f"0:v:{streamidx}",
                format="rawvideo",
                pix_fmt="gray" if args.greyscale else "rgb24",
                loglevel="warning",
            )
            .compile()
        )
        log.info(f"compiled ffmpeg argv: {argv}")

        bufsize = (framesize + 1) * args.buffer_size
        log.info(
            f"max stack size: {args.max_stack_size}",
            f"buffer size: {args.buffer_size}",
            f"real buffer size: {bufsize}",
            f"strict stack: {args.strict_stack}",
            f"max frame count: {args.max_frame_count}",
        )

        pipe = sp.Popen(argv, stdout=sp.PIPE, bufsize=bufsize)

        # TODO: dynamically adjust size of stack depending on memory usage and computation time
        stack = []
        frameidx = 0
        while (
            args.max_frame_count == 0 or frameidx < args.max_frame_count
        ) and pipe.stdout.readable():
            rawframe = pipe.stdout.read(framesize)
            frame = np.frombuffer(rawframe, dtype=np.uint8)

            stacksize = len(stack)
            if args.max_stack_size != 0 and stacksize == args.max_stack_size:
                stack.pop(0)
            else:
                stacksize += 1
            stack.append(frame)
            frameidx += 1

            if stacksize < 2 or (
                args.strict_stack and stacksize != args.max_stack_size
            ):
                continue

            # TODO: implement joint entropy. make joint entropy default (but
            #       optional) via argparse. convert to greyscale until
            #       implemented joint entropy is implemented.
            entropy = delentropy.variation(args, stack)[0]
            log.info(
                f"entropy of frames {frameidx - stacksize + 1}-{frameidx + 1}: {entropy}"
            )


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
