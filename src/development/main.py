#!/usr/bin/env python3
# See `./main.py --help`.


from multiprocessing.queues import Empty as QueueEmptyError
from operator import itemgetter
import json
import multiprocessing as mp
import os
import sys
import time

from tensorflow.compat.v1 import ConfigProto, InteractiveSession
import tensorflow as tf

from argparser import getargs
from log import loginfo
from util import basename
import consts
import workers


__author__ = "Berke Kocaoğlu"
__version__ = "0.1.0"


def configuretf():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True

    return (config, InteractiveSession(config=config))


def deploysegment(files, devs):
    loginfo("Deploying segmentation workers.")

    devnames = [
        *[
            f"/TPU:{tpuidx}"
            for tpuidx, _ in enumerate(
                filter(lambda dev: dev.device_type == "TPU", devs)
            )
        ],
        *[
            f"/GPU:{gpuidx}"
            for gpuidx, _ in enumerate(
                filter(lambda dev: dev.device_type == "GPU", devs)
            )
        ],
        *[
            f"/CPU:{cpuidx}"
            for cpuidx, _ in enumerate(
                filter(lambda dev: dev.device_type == "CPU", devs)
            )
        ],
    ]

    inqueue = mp.Queue()
    for i, fl in enumerate(files):
        inqueue.put((i, fl))

    outqueue = mp.Queue()

    # TODO: use all devices simultaneously
    for d in devnames:
        workers.segment(d, inqueue, outqueue)

    loginfo(f"Total files processed: {outqueue.qsize()}")

    segresults = []
    while True:
        try:
            i = outqueue.get(timeout=consts.mptimeout)
        except QueueEmptyError:
            break

        segresults.append(i)

    loginfo(f"Total results parsed: {len(segresults)}")

    return segresults


def deployentropy(files, cpucount, segresults):
    loginfo(
        f"Deploying entropy worker{'s' if cpucount > 1 or cpucount == 0 else ''}.",
        f"Received {len(segresults)} segmentations, ignoring.",
    )

    with mp.Pool(cpucount) as p:
        results = p.starmap(
            workers.entropy,
            [(i, fl, segresults[i]) for i, fl in enumerate(files)],
        )

    return results


def main():
    files = args.files

    if not files:
        files = sys.stdin.read().split("\0" if args.zero_terminated else "\n")
        if not files[-1]:
            files.pop()

    loginfo("Collecting system information.")

    cpucount = os.cpu_count()
    devs = tf.config.get_visible_devices()
    gpucount = len(list(filter(lambda dev: dev.device_type == "GPU", devs)))
    tpucount = len(list(filter(lambda dev: dev.device_type == "TPU", devs)))

    config, _ = configuretf()

    loginfo(
        "\n\t".join(
            [
                "Using parameters:",
                f"kernel size: {args.kernel_size}",
                f"sigma: {args.sigma}",
                f"mu: {args.mu}",
                "argv: '" + "' '".join(sys.argv) + "'",
                f"cpu count: {cpucount}",
                f"gpu count: {gpucount}",
                f"tpu count: {tpucount}",
                f"tensorflow version: {tf.__version__}",
                f"tensorflow visible devices: {devs}",
                "tensorflow config:\n\t\t" + "\n\t\t".join(str(config).splitlines()),
            ]
        ),
        "Processing files.",
    )

    timepassed = time.monotonic()
    cputimepassed = time.process_time()

    segresults = deploysegment(files, devs)
    segresults.sort(key=itemgetter(0))

    entresults = deployentropy(files, cpucount, segresults)
    entresults = list(filter(lambda x: x is not None, entresults))

    cputimepassed = time.process_time() - cputimepassed
    timepassed = time.monotonic() - timepassed

    loginfo(
        f"Processed {len(files)} files in {cputimepassed:.6f}s "
        + f"process time and {timepassed:.6f}s real time.",
        "Setting up JSON data.",
    )

    utctime = time.gmtime()

    resultjson = {
        "info": {
            "description": "Image Entropy",
            "url": "https://github.com/XPhyro/image-entropy",
            "version": __version__,
            "year": utctime.tm_year,
            "contributor": __author__,
            "date_created": time.strftime("%Y/%m/%d", utctime),
        },
        "licenses": [
            {
                "url": "placeholder",
                "id": 1,
                "name": "placeholder",
            },
        ],
        "images": [
            {
                "id": ifl[0],
                "license": 1,
                "coco_url": "placeholder",
                "flickr_url": "placeholder",
                "width": r["segmentation"]["size"][0],
                "height": r["segmentation"]["size"][1],
                "file_name": basename(ifl[1]),
                "date_captured": "1970-01-01 02:00:00",
            }
            for ifl, r in zip(enumerate(files), entresults)
        ],
        "annotations": entresults,
        "categories": [
            {
                "supercategory": "entropy",
                "id": 1,
                "name": "high_entropy",
            },
        ],
    }

    loginfo("Generating JSON file.")

    with open("cocoout.json", "w", encoding="utf-8") as f:
        json.dump(
            resultjson,
            f,
            ensure_ascii=False,
            indent=4,
            separators=[", ", ": "],
        )

    loginfo("Done.")


if __name__ == "__main__":
    args = getargs()
    main()
