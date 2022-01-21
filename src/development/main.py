#!/usr/bin/env python3.7
# See `./main.py --help`.


from datetime import datetime as dt
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

from scipy.ndimage.filters import gaussian_filter
import cv2 as cv
import numpy as np
import scipy.stats as stats

from pixellib.instance import instance_segmentation
from pycocotools import mask as coco


__author__ = "Berke Kocaoğlu"
__version__ = "0.1.0"

__execname__ = sys.argv[0][sys.argv[0].rfind("/") + 1 :]


def _log(*args, file=os.devnull, **kwargs):
    print(f"{__execname__} [{dt.now()}]: ", file=file, end="")
    print(*args, file=file, **kwargs)


def loginfo(*args, **kwargs):
    _log(*args, file=sys.stdout, **kwargs)


def logerr(*args, **kwargs):
    _log(*args, file=sys.stderr, **kwargs)


def makedirs(path):
    try:
        os.makedirs(path)
    except FileExistsError as err:
        if not os.path.isdir(path):
            raise err


def parseargs():
    global args

    parser = argparse.ArgumentParser(
        description="Find ROIs in images and write to file."
    )

    requiredgroup = parser.add_argument_group("required arguments")
    optionalgroup = parser.add_argument_group("optional arguments")

    optionalgroup.add_argument(
        "-a",
        "--async",
        help=f"enable experimental GPU-CPU simultaneity.",
        action="store_true",
    )

    optionalgroup.add_argument(
        "-f",
        "--infer-speed",
        help=f"speed to use. (default: average)",
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

    optionalgroup.add_argument(
        "-M",
        "--mu",
        help="mu value to be used in distribution thresholding. (default: 0.99)",
        type=argtypeunit,
        default=0.99,
    )

    requiredgroup.add_argument(
        "-m",
        "--model",
        help="tensorflow instance segmentation model to use.",
        required=True,
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

    args = parser.parse_args()


def argtypekernelsize(val):
    ival = int(val)
    if ival <= 1 or ival % 2 != 1:
        raise argparse.ArgumentTypeError(f"{ival} is not a valid kernel size.")
    return ival


def argtypeunit(val):
    uval = float(val)
    if not (0.0 < uval < 1.0):
        raise argparse.ArgumentTypeError(f"{uval} is not a valid mu value.")
    return uval


def argtypeposfloat(val):
    fval = float(val)
    if fval <= 0:
        raise argparse.ArgumentTypeError(f"{fval} must be a positive real.")
    return fval


def cpumain(data):
    idx, fl = data
    idx += 1

    loginfo(f"CPU: Processing file {idx} - {fl}")

    ### read

    inputimg = cv.imread(fl)

    if inputimg is None:  # do not use `not inputimg` for compatibility with arrays
        logerr(f"Could not read file {fl}.")
        return

    greyimg = cv.cvtColor(inputimg, cv.COLOR_BGR2GRAY)

    assert greyimg.dtype == np.uint8, "image channel depth must be 8 bits"

    # prevent over/underflows during computation
    greyimg = greyimg.astype(np.int64)

    ### compute

    grad = np.gradient(greyimg)
    fx = grad[0].astype(int)
    fy = grad[1].astype(int)

    grad = fx + fy

    jrng = np.max([np.max(np.abs(fx)), np.max(np.abs(fy))])
    assert jrng <= 255, "J must be in range [-255, 255]"

    if args.save_images:
        kernshape = (args.kernel_size,) * 2
        kerngrad = np.einsum(
            "ijkl->ij",
            np.lib.stride_tricks.as_strided(
                grad,
                tuple(np.subtract(grad.shape, kernshape) + 1) + kernshape,
                grad.strides * 2,
            ),
        )

    roigrad = np.abs(grad)
    roigradflat = roigrad.flatten()
    mean = np.mean(roigrad)
    roigradbound = (
        mean,
        *stats.t.interval(
            args.mu, len(roigradflat) - 1, loc=mean, scale=stats.sem(roigradflat)
        ),
    )
    roigrad[roigrad < roigradbound[2]] = 0
    roigrad[np.nonzero(roigrad)] = 255

    roigradblurred = gaussian_filter(roigrad, sigma=args.sigma)
    roigradblurred[np.nonzero(roigradblurred)] = 255

    if args.save_images:
        roikerngrad = np.abs(grad)
        roikerngradflat = roikerngrad.flatten()
        mean = np.mean(roigrad)
        roikerngradbound = (
            mean,
            *stats.t.interval(
                args.mu,
                len(roikerngradflat) - 1,
                loc=mean,
                scale=stats.sem(roikerngradflat),
            ),
        )
        roikerngrad[roikerngrad < roikerngradbound[2]] = 0
        roikerngrad[np.nonzero(roikerngrad)] = 255

        roikerngradblurred = gaussian_filter(roikerngrad, sigma=args.sigma)
        roikerngradblurred[np.nonzero(roikerngradblurred)] = 255

    entmasksource = roigradblurred
    entmask = np.asfortranarray(roigradblurred).astype(np.uint8)

    if args.save_images:
        entmaskcolour = cv.cvtColor(entmasksource.astype(np.uint8), cv.COLOR_GRAY2BGR)
        entmaskcolour[:, :, 0:2] = 0
        overlay = np.bitwise_or(entmaskcolour, inputimg)

        results = (
            (grad, "gradient"),
            (kerngrad, "gradient-convolved"),
            (roigrad, "roi"),
            (roigradblurred, "roi-blurred"),
            (roikerngrad, "roi-convolved"),
            (roikerngradblurred, "roi-convolved-blurred"),
            (entmask, "coco-mask"),
            (overlay, "coco-mask-overlayed"),
        )

        parentdir = f"results/{fl[fl.rfind('/') + 1 :]}"

        makedirs(parentdir)

        for r in results:
            data, name = r
            path = f"{parentdir}/{name}.png"
            if os.path.exists(path):
                os.remove(path)
            cv.imwrite(path, data)

    segmentation = coco.encode(entmask)
    size = segmentation["size"]
    counts = list(segmentation["counts"])
    # area = float(np.count_nonzero(entmasksource))
    area = float(coco.area(segmentation))
    bbox = list(coco.toBbox(segmentation))

    ret = {
        "id": idx + 1000000000,
        "category_id": 1,
        "iscrowd": 1,
        "segmentation": {
            "size": size,
            "counts": counts,
        },
        "image_id": idx,
        "area": area,
        "bbox": bbox,
    }

    loginfo(f"CPU: Done processing file {idx} - {fl}")

    return ret


def gpumain(files):
    loginfo(f"GPU: Initialising segmenter with {args.infer_speed} speed.")
    segmenter = instance_segmentation(infer_speed=args.infer_speed)
    loginfo(f"GPU: Loading model {args.model}.")
    segmenter.load_model(args.model)

    for i, fl in enumerate(files):
        loginfo(f"GPU: Processing {i} - {fl}")

        parentdir = f"results/{fl[fl.rfind('/') + 1 :]}"

        makedirs(parentdir)

        segmenter.segmentImage(
            fl,
            output_image_name=f"{parentdir}/segmentation.png",
            show_bboxes=True,
        )

        loginfo(f"GPU: Done processing file {i} - {fl}")


def main():
    parseargs()

    files = args.files
    if not files or not len(files):
        files = sys.stdin.read().split("\0" if args.zero_terminated else "\n")
        if not files[-1]:
            files.pop()

    cpucount = os.cpu_count()
    # processcount = (cpucount * 13) // 12
    processcount = cpucount

    loginfo(
        "\n\t".join(
            [
                "Using parameters:",
                f"maximum process count: {processcount}",
                f"kernel size: {args.kernel_size}",
                f"sigma: {args.sigma}",
                f"mu: {args.mu}",
            ]
        )
    )

    loginfo(f"Processing files.")

    timepassed = time.time()
    cputimepassed = time.process_time()

    loginfo(
        "Deploying CPU process"
        + f"{'es' if processcount > 1 or processcount == 0 else ''}."
    )
    with mp.Pool(processcount) as p:
        if args.async:
            results = p.map_async(cpumain, enumerate(files))
        else:
            results = p.map(cpumain, enumerate(files))

    loginfo("Deploying GPU process.")
    gpumain(files)

    if args.async:
        loginfo("GPU finished, checking on CPU.")

        if not results.ready():
            loginfo("CPU is not finished, waiting.")
            # results.wait()

        results = results.get()
        loginfo("CPU is finished.")

    cputimepassed = time.process_time() - cputimepassed
    timepassed = time.time() - timepassed

    loginfo(
        f"Processed {len(files)} files in {cputimepassed:.9g}s "
        + f"process time and {timepassed:.9g}s real time."
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
    main()
