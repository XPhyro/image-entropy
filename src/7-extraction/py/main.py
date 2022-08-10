#!/usr/bin/env python3


from copy import deepcopy
from operator import itemgetter
import argparse
import json
import os
import subprocess as sp
import sys
import time

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

    reqlst = []
    if len(args.files) == 0:
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


def parserefs():
    global refs
    global refpath

    log.info("parsing reference cache")

    refdir = (
        os.environ.get("XDG_CACHE_HOME") or f"{os.environ['HOME']}/.cache"
    ) + "/ivmer/lava-lamp/ndim-entropy"
    reffl = "reference.json"
    refpath = f"{refdir}/{reffl}"

    if not os.path.exists(refpath):
        log.info("creating empty reference cache")
        if not os.path.exists(refdir):
            os.makedirs(refdir)
        with open(refpath, "w", encoding="utf-8") as fl:
            fl.write("{}\n")

    log.info("loading reference cache")
    with open(refpath, "r", encoding="utf-8") as fl:
        refs = json.load(fl)


def getref(shape):
    shapeparams = ", ".join(np.char.mod("%d", shape))
    argparams = ", ".join(
        str(i)
        for i in [
            args.abstract_entropy,
            args.buffer_size,
            args.cycle_rgb,
            args.stack_is_stream,
            args.greyscale,
            args.binary_height,
            args.light_variation,
            args.stack_period,
            args.stack_modulus,
            args.max_frame_count,
            args.skip_period,
            args.strict_stack,
            args.max_stack_size,
            args.binary_width,
            args.binary_color,
        ]
    )
    hashstr = f"[{shapeparams}], [{argparams}]"
    log.info(f"searching for a suitable reference for {hashstr}")

    if args.pi_path:
        log.info("trying pi reference")
        if (ref := refs.get(f"Pi: {hashstr}")) is None:
            if args.pi_path and (ref := calcref(shape, args.pi_path)) is not None:
                refs[f"Pi: {hashstr}"] = ref
                overwriterefs()
    if (not args.pi_path or ref is None) and (
        ref := refs.get(f"urandom: {hashstr}")
    ) is None:
        log.info("trying urandom reference")
        if (ref := calcref(shape, "/dev/urandom")) is not None:
            refs[f"urandom: {hashstr}"] = ref
            overwriterefs()
    log.info(f"found reference: {ref}")
    return ref


def overwriterefs():
    log.info("updating reference cache")
    with open(refpath, "w", encoding="utf-8") as fl:
        json.dump(
            refs,
            fl,
            ensure_ascii=False,
            indent=4,
            separators=[", ", ": "],
        )


def calcref(shape, path):
    max_frame_count = args.max_frame_count
    strict_stack = args.strict_stack
    treat_binary = args.treat_binary
    binary_height = args.binary_height
    binary_width = args.binary_width

    args.max_frame_count = args.max_stack_size
    args.strict_stack = True
    args.treat_binary = True
    args.binary_height = shape[0]
    args.binary_width = shape[1]

    entropies = getentropies(path, False)

    args.max_frame_count = max_frame_count
    args.strict_stack = strict_stack
    args.treat_binary = treat_binary
    args.binary_height = binary_height
    args.binary_width = binary_width

    if entropies is not None:
        return np.mean(entropies)

    return None


def getpipes(filename):
    log.info(f"log.noout: {log.noout}")
    log.info(f"log.infoenabled: {log.infoenabled}")
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
        log.info(f"compiled reader argv: {argv}")

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
            log.info(f"compiled reader argv: {argv}")

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

    return (nstreams, shape, framesize, bufsize, pipes)


def getentropies(filename, normalise=True):
    nstreams, shape, framesize, _, pipes = getpipes(filename)
    if nstreams == 0:
        log.err("no video streams found in file, skipping")
        return None
    if normalise:
        ref = getref(shape)
        if ref is None:
            log.warn("no reference found, not normalising")
            normalise = False

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
        pipe = None
        while pipe is None:
            pipe = pipes[pipeidx]
            if not pipe.stdout.readable():
                log.info(f"pipe {pipe} is not readable, removing")
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
        if normalise:
            entropy /= ref
        entropies.append(entropy)
        log.info(
            f"entropy of frames {frameidx - stacksize + 1}-{frameidx + 1} ({stacksize}): {entropy}"
        )

    return entropies


def deployextractors(shape, framesize, bufsize, normalise, ref):
    cpucount = os.sched_getaffinity(0)
    log.info(f"available cpus: {cpucount}", f"available cpu count: {len(cpucount)}")

    pipes = []
    for idx, cpu in enumerate(cpucount):
        argv = deepcopy(sys.argv)
        argv[0] = f"{argv[0][:argv[0].rfind('/')]}/extract.py"
        argv.append(str(idx))
        argv.append(str(cpu))
        argv.append(str(shape))
        argv.append(str(framesize))
        argv.append(str(bufsize))
        argv.append(str(normalise))
        argv.append(str(ref))
        log.info(f"compiled extractor argv: {argv}")

        pipes.append(
            sp.Popen(
                argv,
                stdin=sp.PIPE,
                stdout=sys.stdout,
                stderr=sys.stderr,
                bufsize=bufsize,
            )
        )

    return pipes


def distributeframes(filename, normalise=True):
    nstreams, shape, framesize, bufsize, inpipes = getpipes(filename)
    if nstreams == 0:
        log.err("no video streams found in file, skipping")
        return None
    if normalise:
        ref = getref(shape)
        if ref is None:
            log.warn("no reference found, not normalising")
            normalise = False
    outpipes = deployextractors(shape, framesize, bufsize, normalise, ref)

    stack = []
    frameidx = 0
    frameskip = 0
    willskip = False
    inpipeidx = 0
    stackidx = 0
    stackskip = 0

    # from now on, stdout is reserved for extractors
    log.infoenabled = False

    while len(outpipes) and (
        args.max_frame_count == 0 or frameidx < args.max_frame_count
    ):
        inpipe = None
        while inpipe is None:
            inpipe = inpipes[inpipeidx]
            if not inpipe.stdout.readable():
                log.err(f"pipe {inpipe} is not readable, removing")
                inpipes.pop(inpipeidx)
                nstreams -= 1
                inpipeidx = inpipeidx % nstreams
            else:
                inpipeidx = (inpipeidx + 1) % nstreams
            if nstreams == 0:
                break
        if inpipe is None:
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

        rawframe = inpipe.stdout.read(framesize)
        if args.skip_period != 0:
            if frameskip != 0:
                frameidx += 1
                willskip = True
            else:
                willskip = False
            frameskip = (frameskip + 1) % args.skip_period
            if willskip:
                continue

        outpipestoremove = []
        for outpipeidx in range(len(outpipes)):
            outpipe = outpipes[outpipeidx]
            if not outpipe.stdin.writable():
                log.err(f"pipe {outpipe} is not writable, removing")
                outpipestoremove.append(outpipeidx)
                continue
            try:
                outpipe.stdin.write(rawframe)
            except BrokenPipeError:
                log.err(f"pipe {outpipe} is broken, removing")
                outpipestoremove.append(outpipeidx)
                continue
        outpipestoremove.reverse()
        for outpipeidx in outpipestoremove:
            outpipes.pop(outpipeidx)


def main():
    global func

    parseargs()

    log.spongeout = args.sponge_out
    log.spongeerr = args.sponge_err
    if args.quiet:
        log.infoenabled = False
        log.warnenabled = False
        log.errenabled = False

    # stdout is reserved for extractors
    log.noout = True

    parserefs()

    delentropy.init(args)

    func = delentropy.variationlight if args.light_variation else delentropy.variation

    for file in args.files:
        distributeframes(file)

    log.dumpcaches()

    return 0


if __name__ == "__main__":
    sys.exit(main())
