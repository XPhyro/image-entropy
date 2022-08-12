#!/usr/bin/env python3


import argparse
import sys

import numpy as np

import delentropy
import log


def parseargs():
    global args
    global extractorindex
    global extractorid
    global shape
    global framesize
    global bufsize
    global normalise
    global ref

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
        "-d",
        "--double-buffer",
        help="double buffer stack",
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
        "-P",
        "--pi-path",
        help="path to binary pi file",
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

    specscount = 7

    reqlst = []
    if len(args.files) < 1 + specscount:
        reqlst.append("FILE")
    if len(reqlst) != 0:
        parser.error(f"the following arguments are required: {', '.join(reqlst)}")

    if args.max_stack_size < 2:
        parser.error("maximum stack size cannot be less than 2")

    if not args.greyscale:
        log.warn(
            "joint entropy is not yet implemented, colour frame assessment will be inaccurate. "
            + "consider using --greyscale."
        )

    extractorindex = int(args.files[-specscount + 0])
    extractorid = int(args.files[-specscount + 1])
    shape = np.fromstring(args.files[-specscount + 2][1:-1], dtype=int, sep=" ")
    framesize = int(args.files[-specscount + 3])
    bufsize = int(args.files[-specscount + 4])
    normalise = bool(args.files[-specscount + 5])
    ref = float(args.files[-specscount + 6])

    args.files = args.files[:-specscount]

    if not args.double_buffer:
        args.strict_stack = True


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


def submitbits(bits):
    sys.stdout.write(bits)


def extractbits():
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
    entropies = []
    while args.max_frame_count == 0 or frameidx < args.max_frame_count:
        stacksize = len(stack)
        if stacksize == args.stack_modulus and stackskip != 0:
            stackskip -= 1
            continue
        stackidx += 1
        if args.stack_period != 0:
            if stackidx % args.stack_period == 0:
                stack.clear()
                stacksize = 0

        rawframe = sys.stdin.buffer.read(framesize)
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

        if args.double_buffer:
            oldstack = [i for i in stack]

        stacksize = len(stack)
        if args.max_stack_size != 0:
            while stacksize >= args.max_stack_size:
                stack.pop(0)
                stacksize -= 1
        stacksize += 1
        stack.append(frame)
        frameidx += 1

        if (
            stacksize < 2
            or (args.strict_stack and stacksize != args.max_stack_size)
            or (args.double_buffer and len(oldstack) != len(stack))
        ):
            continue

        # TODO: implement joint entropy. make joint entropy default (but
        #       optional) via argparse.
        entropy, _ = func(args, stack)
        normalentropy = entropy
        if normalise:
            normalentropy /= ref

        log.warn(
            f"entropy of frames {frameidx - stacksize + 1}-{frameidx + 1} ({stacksize}): {entropy}",
            f"normalised entropy of frames {frameidx - stacksize + 1}-{frameidx + 1} ({stacksize}): {normalentropy}",
        )

        entropies.append(normalentropy)
        log.warn(
            f"current estimate for normalised entropy: {np.mean(entropies)} ± {np.std(entropies)}"
        )

        if args.double_buffer:
            xor = np.bitwise_xor(stack, oldstack).flatten().astype(np.uint8)
        else:
            xor = stack[0]
            for i in stack[1:]:
                xor = np.bitwise_xor(xor, i)
            xor = xor.flatten().astype(np.uint8)
        xorsum = np.sum(xor)
        xormean = np.mean(xor)
        normalxorsum = xorsum / 8e8

        log.warn(f"xor shape: {xor.shape}")
        log.warn(f"xor sum: {xorsum}")
        log.warn(f"normalised xor sum: {normalxorsum}")
        log.warn(f"mean xor: {xormean}")
        log.warn(f"normalised mean xor: {xormean / (2**7 - 0.5)}")

        # TODO: ideally we would prioritise the first bits,
        #       but using a percentage of shuffled bytes is easier.
        bytescount = int(normalentropy * np.prod(xor.shape))
        np.random.shuffle(xor)
        entropybytes = xor[: bytescount - 1]

        log.raw(entropybytes.tobytes())

    return entropies


def main():
    global func

    parseargs()

    log.spongeout = args.sponge_out
    log.spongeerr = args.sponge_err
    if args.quiet:
        log.infoenabled = False
        log.warnenabled = False
        log.errenabled = False

    # stdout is reserved for entropy bits
    log.infoenabled = False
    log.noout = True

    log.init(extractorid)
    log.warn(f"extractor initialised with id: {extractorid}")

    delentropy.init(args)

    func = delentropy.variationlight if args.light_variation else delentropy.variation

    extractbits()

    log.dumpcaches()

    return 0


if __name__ == "__main__":
    sys.exit(main())
