from multiprocessing.queues import Empty as QueueEmptyError

from scipy import stats
from scipy.ndimage.filters import gaussian_filter
import cv2 as cv
import numpy as np
import tensorflow as tf

from pixellib.semantic import semantic_segmentation
from pycocotools import mask as coco

from argparser import getargs
from log import loginfo, logerr
import consts
import util


def entropy(idx, fl, segmentation):
    parentdir = f"results/segmentation/{util.basename(fl)}"

    loginfo(f"CPU: Processing file {idx} - {fl}")

    ### read

    inputimg = cv.imread(fl)

    if inputimg is None:  # do not use `not inputimg` for compatibility with arrays
        logerr(f"Could not read file {fl}.")
        return None

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

    entmasksource = roigrad
    entmask = np.asfortranarray(entmasksource).astype(np.uint8)

    # PixelLib's semantic segmentation is bugged in multiple ways:
    #     1. It does not return class masks correctly.
    #     2. It does not retain class colour order.
    #
    # segdata.keys() := dict_keys(['class_ids', 'class_names', 'class_colors', 'masks', 'ratios'])
    # segmap.shape = (w, h, 3)
    _, (segdata, segmap) = segmentation

    loginfo(
        f"{idx} - {fl} - segmap.shape = {segmap.shape}",
        f"{idx} - {fl} - grad.shape = {grad.shape}",
        f"{idx} - {fl} - segdata.keys() = {segdata.keys()}",
        f"{idx} - {fl} - segdata['class_names'] = {segdata['class_names']}",
    )

    # The following is a workaround for both (1) and (2).
    objmasks = {}
    for rowidx, row in enumerate(segmap):
        for colidx, elem in enumerate(row):
            elemint = elem[0] + (elem[1] << 8) + (elem[2] << 16)
            if elemint not in objmasks.keys():
                objmasks[elemint] = np.zeros(grad.shape).astype(bool)
            else:
                objmasks[elemint][rowidx][colidx] = True
    objimgs = {}
    for key, val in objmasks.items():
        obj = np.copy(segmap)
        obj[np.invert(val)] = [0, 0, 0]

        objimgs[key] = obj

    objentmaps = {}
    for key, mask in objmasks.items():
        objentmaps[key] = (np.multiply(fx, mask), np.multiply(fy, mask))
    objents = {}
    for key, (objfx, objfy) in objentmaps.items():
        hist, _, _ = np.histogram2d(
            objfx.flatten(),
            objfy.flatten(),
            bins=256,
            range=[[-jrng, jrng], [-jrng, jrng]],
        )

        deldensity = hist / np.sum(hist)
        deldensity = deldensity * -np.ma.log2(deldensity)
        ent = np.sum(deldensity) / 2

        objents[key] = ent

    loginfo(f"{idx} - {fl} - entropies = {objents}")

    ret = []
    for entidx, roiobj in enumerate(objents.keys()):
        objentmasksource = grad
        objentmasktopsource = roigrad

        objentmaskimg = (
            np.logical_and(objentmasksource, objmasks[roiobj]).astype(np.uint8) * 255
        )
        objentmasktopimg = (
            np.logical_and(objentmasktopsource, objmasks[roiobj]).astype(np.uint8) * 255
        )

        objentmask = np.asfortranarray(objentmaskimg)

        if args.save_images:
            entmaskcolour = cv.cvtColor(
                entmasksource.astype(np.uint8), cv.COLOR_GRAY2BGR
            )
            entmaskcolour[:, :, 0:2] = 0
            entmaskoverlay = np.bitwise_or(entmaskcolour, inputimg)

            objentmaskcolour = cv.cvtColor(objentmaskimg, cv.COLOR_GRAY2BGR)
            objentmaskcolour[:, :, 0:2] = 0
            objentmaskoverlay = np.bitwise_or(objentmaskcolour, inputimg)

            objentmasktopcolour = cv.cvtColor(objentmasktopimg, cv.COLOR_GRAY2BGR)
            objentmasktopcolour[:, :, 0:2] = 0
            objentmasktopoverlay = np.bitwise_or(objentmasktopcolour, inputimg)

            results = (
                (grad, "gradient"),
                (kerngrad, "gradient-convolved"),
                (roigrad, "roi"),
                (roigradblurred, "roi-blurred"),
                (roikerngrad, "roi-convolved"),
                (roikerngradblurred, "roi-convolved-blurred"),
                (entmask, "coco-mask"),
                (entmaskoverlay, "coco-mask-overlayed"),
                (objimgs[roiobj], f"roi-mask-{roiobj}-{objents[roiobj]}"),
                (objentmaskimg, "coco-obj-mask"),
                (objentmasktopimg, "coco-obj-mask-top"),
                (objentmaskoverlay, "coco-obj-mask-overlayed"),
                (objentmasktopoverlay, "coco-obj-mask-top-overlayed"),
            )

            util.makedirs(parentdir)

            for r in results:
                data, name = r
                path = f"{parentdir}/{name}.png"
                cv.imwrite(path, data)

            util.makedirs(f"{parentdir}/masks")
            for key, obj in objimgs.items():
                cv.imwrite(f"{parentdir}/masks/mask-{key}.png", obj)

        encodedmask = coco.encode(objentmask)
        size = encodedmask["size"]
        # counts = list(encodedmask["counts"])
        counts = encodedmask["counts"]
        # area = float(np.count_nonzero(entmasksource))
        area = float(coco.area(encodedmask))
        bbox = list(coco.toBbox(encodedmask))

        ret.append(
            {
                # TODO: make this better. currently we only support up to 1 million
                #       images and the index can be greater than 2**32.
                "id": (1 + idx) * 1000000 + entidx + 1,
                "category_id": 1,
                "iscrowd": 1,
                "segmentation": {
                    "size": size,
                    "counts": counts.decode("utf-8"),
                },
                "image_id": idx + 1,
                "area": area,
                "bbox": bbox,
                "entropy": objents[roiobj],
            }
        )

    loginfo(f"CPU: Done processing file {idx} - {fl}")

    return ret


def segment(devname, inqueue, outqueue):
    with tf.device(devname):
        loginfo(f"{devname}: Initialising semantic segmenter.")
        semanticsegmenter = semantic_segmentation()

        if args.semantic_model_ade20k:
            loginfo(f"{devname}: Loading model {args.semantic_model_ade20k}.")
            semanticsegmenter.load_ade20k_model(args.semantic_model_ade20k)

            def getsegmented(fl, parentdir):
                semanticsegmenter.segmentAsAde20k(
                    fl,
                    output_image_name=f"{parentdir}/segmentation-semantic-ade20k.png",
                    overlay=False,
                )

        else:
            loginfo(f"{devname}: Loading model {args.semantic_model_pascalvoc}.")
            semanticsegmenter.load_pascalvoc_model(args.semantic_model_pascalvoc)

            def getsegmented(fl, parentdir):
                semanticsegmenter.segmentAsPascalvoc(
                    fl,
                    output_image_name=f"{parentdir}/segmentation-semantic-pascalvoc.png",
                    overlay=False,
                )

        while True:
            try:
                idx, fl = inqueue.get(timeout=consts.mptimeout)
            except QueueEmptyError:
                return

            loginfo(f"{devname}: Processing {idx} - {fl}")

            parentdir = f"results/segmentation/{util.basename(fl)}"

            util.makedirs(parentdir)

            outqueue.put((idx, getsegmented(fl, parentdir)))

            loginfo(
                f"{devname}: Done processing file {idx} - {fl}",
                f"{devname}: Total files processed: {outqueue.qsize()}",
            )


args = getargs()
