#!/usr/bin/env python3
# See `./main.py --help`.


from datetime import datetime as dt
import json
import multiprocessing as mp
import os
import sys
import time

from tensorflow.compat.v1 import ConfigProto, InteractiveSession
import tensorflow as tf

from argparser import getargs
from log import loginfo, logerr
import workers


__author__ = "Berke Kocaoğlu"
__version__ = "0.1.0"


def configuretf():
    # tf.config.set_visible_devices([], device_type="gpu")
    # tf.debugging.set_log_device_placement(True)

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True

    return (config, InteractiveSession(config=config))


def main():
    files = args.files

    if not files or not len(files):
        files = sys.stdin.read().split("\0" if args.zero_terminated else "\n")
        if not files[-1]:
            files.pop()

    cpucount = os.cpu_count()
    devs = tf.config.get_visible_devices()
    gpucount = len(list(filter(lambda dev: dev.device_type == "GPU", devs)))

    config, session = configuretf()

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
                f"tensorflow version: {tf.__version__}",
                f"tensorflow visible devices: {devs}",
                "tensorflow config:\n\t\t" + "\n\t\t".join(str(config).splitlines()),
            ]
        ),
        "Processing files.",
    )

    timepassed = time.monotonic()
    cputimepassed = time.process_time()

    loginfo("Deploying segmentation workers.")

    devnames = [
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
    segqueue = mp.Queue()
    segresults = mp.Queue()

    loginfo(
        "Deploying entropy worker" + f"{'es' if cpucount > 1 or cpucount == 0 else ''}."
    )
    with mp.Pool(cpucount) as p:
        results = p.map(workers.entropy, list(enumerate(files)))

    cputimepassed = time.process_time() - cputimepassed
    timepassed = time.monotonic() - timepassed

    loginfo(
        f"Processed {len(files)} files in {cputimepassed:.6f}s "
        + f"process time and {timepassed:.6f}s real time."
    )

    results = list(filter(lambda x: x is not None, results))

    loginfo(f"Setting up JSON data.")

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
                "file_name": ifl[1][ifl[1].rfind("/") + 1 :],
                "date_captured": "1970-01-01 02:00:00",
            }
            for ifl, r in zip(enumerate(files), results)
        ],
        "annotations": results,
        "categories": [
            {
                "supercategory": "entropy",
                "id": 1,
                "name": "high_entropy",
            },
        ],
    }

    loginfo(f"Generating JSON file.")

    with open("cocoout.json", "w", encoding="utf-8") as f:
        json.dump(
            resultjson,
            f,
            ensure_ascii=False,
            indent=4,
            separators=[", ", ": "],
        )

    loginfo(f"Done.")


if __name__ == "__main__":
    args = getargs()
    main()
