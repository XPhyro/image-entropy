#!/usr/bin/env python3


import sys

import numpy as np

import log


def processvideo():
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


def main():
    return 0


if __name__ == "__main.py__":
    sys.exit(main())
