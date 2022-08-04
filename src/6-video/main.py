#!/usr/bin/env python3


from operator import itemgetter
import argparse
import subprocess as sp
import sys

import ffmpeg as ffm
import numpy as np

import delentropy
import log


def parseargs():
    global args

    parser = argparse.ArgumentParser(description="Compute and display image entropy.")

    parser.add_argument(
        "-a",
        "--abstract-entropy",
        help="use abstract entropy instead of strict information entropy",
        action="store_true",
    )

    parser.add_argument(
        "-B",
        "--treat-binary",
        help="treat given files as binary instead of video. if given, -H and -W are required.",
        action="store_true",
    )

    parser.add_argument(
        "-b",
        "--buffer-size",
        help="count of frames to buffer. default is 4.",
        type=argtypepint,
        default=4,
    )

    parser.add_argument(
        "-C",
        "--cycle-rgb",
        help="cycle usage of r, g and b channels",
        action="store_true",
    )

    parser.add_argument(
        "-c",
        "--stack-is-stream",
        help="sync stack period, stack modulus and maximum stack size to stream count. overrides -M, -m and -s.",
        action="store_true",
    )

    parser.add_argument(
        "-G",
        "--gpu",
        help="use GPU",
        action="store_true",
    )

    parser.add_argument(
        "-g",
        "--greyscale",
        help="convert frames to greyscale",
        action="store_true",
    )

    parser.add_argument(
        "-H",
        "--binary-height",
        help="height of the video if -B is given. default is 1200.",
        type=argtypepint,
        default=1200,
    )

    parser.add_argument(
        "-L",
        "--light-variation",
        help="use lighter delentropy variation. uses less processing and memory.",
        action="store_true",
    )

    parser.add_argument(
        "-M",
        "--stack-period",
        help="number of frames to wait until sliding the stack. 0 is disabled. default is 0.",
        type=argtypeuint,
        default=0,
    )

    parser.add_argument(
        "-m",
        "--stack-modulus",
        help="number of frames to slide to construct a new stack. 0 is disabled. if non-zero, frame indices shown will be wrong. default is 0.",
        type=argtypeuint,
        default=0,
    )

    parser.add_argument(
        "-n",
        "--max-frame-count",
        help="limit number of frames processed per stream per file. 0 is unlimited. default is 0.",
        type=argtypeuint,
        default=0,
    )

    parser.add_argument(
        "-p",
        "--skip-period",
        help="skip every nth frame. 0 is no skipping. if non-zero, frame indices shown will be wrong. default is 0.",
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
        help="maximum size of the frame stack in count of frames. 0 is unlimited. effective maximum size may be smaller than the given maximum size. default is 8.",
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
        "-W",
        "--binary-width",
        help="width of the video if -B is given. default is 1920.",
        type=argtypepint,
        default=1920,
    )

    parser.add_argument(
        "-Q",
        "--binary-color",
        help="enable color if -B is given",
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
    if args.max_stack_size < 2:
        log.err("maximum stack size cannot be less than 2")
        sys.exit(1)


def argtypeuint(val):
    ival = int(val)
    if ival < 0:
        raise argparse.ArgumentTypeError("argument must be a non-negative integer.")
    return ival


def argtypepint(val):
    ival = int(val)
    if ival <= 0:
        raise argparse.ArgumentTypeError("argument must be a positive integer.")
    return ival


def pipevideo(filename):
    log.info(f"reading {filename}")

    pipes = []

    if args.treat_binary:
        nstreams = 1
        shape = (args.binary_height, args.binary_width)
        if args.binary_color:
            shape = (*shape, 3)
        framesize = int(np.prod(shape, dtype=int))
        log.info(f"shape: {shape}", f"frame size: {framesize}")

        argv = ["cat", "--", filename]
        log.info(f"compiled argv: {argv}")

        bufsize = (framesize + 1) * args.buffer_size
        log.info(
            f"max stack size: {args.max_stack_size}",
            f"buffer size: {args.buffer_size}",
            f"real buffer size: {bufsize}",
            f"strict stack: {args.strict_stack}",
            f"max frame count: {args.max_frame_count}",
        )

        pipes.append(sp.Popen(argv, stdout=sp.PIPE, bufsize=bufsize))
        log.info("successfully opened pipe")
    else:
        streams = ffm.probe(filename, select_streams="v")["streams"]
        nstreams = len(streams)
        if nstreams == 0:
            return (0, None, None, None)

        log.info(f"found {nstreams} {'stream' if nstreams == 1 else 'streams'}")
        for i, stream in enumerate(streams):
            streamidx = stream["index"]
            streamhidx = streamidx + 1

            log.info(f"processing stream {streamhidx}/{nstreams}")

            log.info(f"total frame count: {stream.get('nb_frames')}")

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
                    map=f"0:v:{i}",
                    format="rawvideo",
                    pix_fmt="gray" if args.greyscale else "rgb24",
                    loglevel="warning",
                )
                .compile()
            )
            log.info(f"compiled argv: {argv}")

            bufsize = (framesize + 1) * args.buffer_size
            log.info(
                f"max stack size: {args.max_stack_size}",
                f"buffer size: {args.buffer_size}",
                f"real buffer size: {bufsize}",
                f"strict stack: {args.strict_stack}",
                f"max frame count: {args.max_frame_count}",
            )

            pipes.append(sp.Popen(argv, stdout=sp.PIPE, bufsize=bufsize))
            log.info("successfully opened pipe")

    return (nstreams, shape, framesize, pipes)


def processvideo(func, filename):
    nstreams, shape, framesize, pipes = pipevideo(filename)
    if nstreams == 0:
        log.err("no video streams found in file, skipping")
        return

    # TODO: dynamically adjust size of stack depending on memory usage and computation time
    stack = []
    frameidx = 0
    frameskip = 0
    willskip = False
    pipeidx = 0
    stackidx = 0
    stackskip = 0
    rgbframe = np.zeros(shape)
    rgbturn = 0
    while args.max_frame_count == 0 or frameidx < args.max_frame_count:
        pipe = None
        while pipe is None:
            pipe = pipes[pipeidx]
            if not pipe.stdout.readable():
                pipes.pop(pipeidx)
                nstreams -= 1
                pipeidx = pipeidx % nstreams
            else:
                pipeidx = (pipeidx + 1) % nstreams
            if nstreams == 0:
                break
        if pipe is None:
            break

        if args.stack_is_stream:
            args.stack_modulus = nstreams
            args.stack_period = nstreams
            args.max_stack_size = nstreams

        stacksize = len(stack)
        if stacksize == args.stack_modulus and stackskip != 0:
            stackskip -= 1
            continue
        stackidx += 1
        if args.stack_period != 0:
            if stackidx % args.stack_period == 0:
                stack.clear()
                stacksize = 0

        rawframe = pipe.stdout.read(framesize)
        if args.skip_period != 0:
            if frameskip != 0:
                frameidx += 1
                willskip = True
            else:
                willskip = False
            frameskip = (frameskip + 1) % args.skip_period
            if willskip:
                continue

        frame = np.frombuffer(rawframe, dtype=np.uint8)
        frame = frame.reshape(shape)

        if args.cycle_rgb:
            rgbframe[:, :, rgbturn] = frame[:, :, rgbturn]
            rgbturn = (rgbturn + 1) % 3
            if rgbturn != 2:
                continue
            frame = rgbframe

        stacksize = len(stack)
        if args.max_stack_size != 0:
            while stacksize >= args.max_stack_size:
                stack.pop(0)
                stacksize -= 1
        stacksize += 1
        stack.append(frame)
        frameidx += 1

        if stacksize < 2 or (args.strict_stack and stacksize != args.max_stack_size):
            continue

        # TODO: implement joint entropy. make joint entropy default (but
        #       optional) via argparse.
        entropy, _ = func(args, stack)
        log.info(
            f"entropy of frames {frameidx - stacksize + 1}-{frameidx + 1} ({stacksize}): {entropy}"
        )


def main():
    parseargs()

    log.spongeout = args.sponge_out
    log.spongeerr = args.sponge_err
    if args.quiet:
        log.infoenabled = False
        log.warnenabled = False
        log.errenabled = False

    delentropy.init(args)

    func = delentropy.variationlight if args.light_variation else delentropy.variation

    for file in args.files:
        processvideo(func, file)

    log.dumpcaches()


if __name__ == "__main__":
    main()
