#!/usr/bin/env python3
# See `./main.py --help`.


import json
import sys

import cv2 as cv
import numpy as np

from pycocotools import mask as coco

from argparser import getargs
from log import loginfo
from util import makedirs


__author__ = "Berke Kocaoğlu"
__version__ = "0.1.0"


def main():
    files = args.files

    if not files:
        files = sys.stdin.read().split("\0" if args.zero_terminated else "\n")
        if not files[-1]:
            files.pop()

    for file in files:
        loginfo(f"Parsing {file}.")

        with open(file, "r", encoding="utf-8") as f:
            cocodict = json.load(f)

        loginfo("Parsing image ID and paths.")

        # do not read at this stage to conserve memory (and support high image counts)
        imgpathsbyid = {}
        imgnamesbyid = {}
        for image in cocodict["images"]:
            imgid = image["id"]
            imgpathsbyid[imgid] = f"{args.image_directory}/{image['file_name']}"
            imgnamesbyid[imgid] = image["file_name"]

        loginfo("Parsing annotations.")

        makedirs(args.result_directory)
        for ann in cocodict["annotations"]:
            annid = ann["id"]
            imgid = ann["image_id"]
            segm = ann["segmentation"]
            size = segm["size"]
            counts = segm["counts"]

            encodedmask = coco.frPyObjects([counts], size[0], size[1])
            mask = coco.decode(encodedmask)

            cv.imwrite(f"{args.result_directory}/{imgnamesbyid[imgid]}", mask * 255)

    loginfo("Done.")


if __name__ == "__main__":
    args = getargs()
    main()
