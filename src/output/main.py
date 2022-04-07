#!/usr/bin/env python3
# See `./main.py --help`.


from base64 import b64decode
import json

import numpy as np

from pycocotools import mask as coco

from argparser import getargs
from log import loginfo


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

        imagepathsbyid = {}
        for image in cocodict["images"]:
            imagepathsbyid[image["id"]] = image["file_name"]

        loginfo("Parsing annotations.")

        annotations = [
            (
                annotation["segmentation"]["size"],
                annotation["segmentation"]["counts"],
                annotation["image_id"],
                annotation["id"],
            )
            for annotation in cocodict["annotations"]
        ]

        for ann in annotations:
            size = ann[0]
            counts = ann[1]
            encodedmask = coco.frPyObjects([counts], size[0], size[1])
            mask = coco.decode(encodedmask)

    loginfo("Done.")


if __name__ == "__main__":
    args = getargs()
    main()
