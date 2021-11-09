#######################################################################################################
# method              | resources                                                                     #
# =================================================================================================== #
# 2d-regional-shannon |                                                                               #
# --------------------------------------------------------------------------------------------------- #
# 2d-gradient         | https://arxiv.org/abs/1609.01117                                              #
# --------------------------------------------------------------------------------------------------- #
# 2d-delentropy       | https://arxiv.org/abs/1609.01117                                              #
#                     | https://github.com/Causticity/sipp                                            #
# --------------------------------------------------------------------------------------------------- #
# 2d-regional-scikit  | https://scikit-image.org/docs/dev/auto_examples/filters/plot_entropy.html     #
#                     | https://scikit-image.org/docs/dev/api/skimage.filters.rank.html               #
# --------------------------------------------------------------------------------------------------- #
# 1d-shannon          |                                                                               #
# --------------------------------------------------------------------------------------------------- #
# 1d-scipy            | https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html #
# --------------------------------------------------------------------------------------------------- #
# 1d-kapur            | https://doi.org/10.1080/09720502.2020.1731976                                 #
#######################################################################################################


from copy import deepcopy as duplicate

from scipy.stats import entropy as spentropy
from skimage.filters.rank import entropy as skentropy
from skimage.morphology import disk as skdisk
import cv2 as cv
import numpy as np

from log import log


def kapur1dv(args, colourimg, greyimg):
    hist = np.histogram(greyimg, bins=255, range=(0, 256))[0]
    cdf = hist.astype(float).cumsum()  # cumulative distribution function
    binrng = np.nonzero(hist)[0][[0, -1]]

    entropymax, threshold = 0, 0
    for i in range(binrng[0], binrng[1] + 1):
        histrng = hist[: i + 1] / cdf[i]
        entropy = -np.sum(histrng * np.ma.log(histrng))

        histrng = hist[i + 1 :]
        histrng = histrng[np.nonzero(histrng)] / (cdf[binrng[1]] - cdf[i])
        entropy -= np.sum(histrng * np.log(histrng))

        if entropy > entropymax:
            entropymax, threshold = entropy, i

    log(f"entropy: {entropy}")
    log(f"threshold: {threshold}")
    log(f"entropy ratio: {entropy / 8.0}")

    entimg = np.where(greyimg < threshold, greyimg, 0)

    return (
        f"{entropy} after 1 iteration",
        colourimg,
        greyimg,
        [(entimg, "Kapur Threshold", ["hasbar"])],
    )


def shannon1d(args, colourimg, greyimg):
    value, counts = np.unique(greyimg.flatten(), return_counts=True)
    entropy = spentropy(counts, base=2)

    log(f"entropy: {entropy}")
    log(f"entropy ratio: {entropy / 8.0}")

    return (entropy, None, None, None)


def delentropy2d(args, colourimg, greyimg):
    ### 1609.01117 page 10

    # $\nabla f(n) \approx f(n) - f(n - 1)$
    fx = greyimg[:, 2:] - greyimg[:, :-2]
    fy = greyimg[2:, :] - greyimg[:-2, :]
    # fix shape
    fx = fx[1:-1, :]
    fy = fy[:, 1:-1]

    # TODO: is this how fx and fy are combined?
    #       it's for plotting and not used in computation anyways,
    #       and it matches the image in the paper.
    grad = fx + fy

    # ensure $-255 \leq J \leq 255$
    jrng = np.max([np.max(np.abs(fx)), np.max(np.abs(fy))])
    assert jrng <= 255, "J must be in range [-255, 255]"

    ### 1609.01117 page 16, eq 17

    hist, edgex, edgey = np.histogram2d(
        fx.flatten(),
        fy.flatten(),
        bins=255,
    )

    ### 1609.01117 page 20, eq 22

    deldensity = hist / np.sum(hist)
    deldensity = deldensity * -np.ma.log2(deldensity)
    entropy = np.sum(deldensity)

    entropy /= 2  # 4.3 Papoulis generalized sampling halves the delentropy

    # TODO: entropy is different from `sipp` and the paper, but very similar
    log(f"entropy: {entropy}")
    log(f"entropy ratio: {entropy / 8.0}")

    # the reference image seems to be bitwise inverted, I don't know why.
    # the entropy doesn't change when inverted, so both are okay in
    # the previous computational steps.
    param_invert = True

    gradimg = np.invert(grad) if param_invert else grad

    return (
        entropy,
        colourimg,
        greyimg,
        [
            (gradimg, "Gradient", ["hasbar"]),
            (deldensity, "Deldensity", ["hasbar"]),
        ],
    )


def delentropynd(args, colourimg, greyimg):
    ### 1609.01117 page 10

    # $\nabla f(n) \approx f(n) - f(n - 1)$
    fx = greyimg[:, 2:] - greyimg[:, :-2]
    fy = greyimg[2:, :] - greyimg[:-2, :]
    # fix shape
    fx = fx[1:-1, :]
    fy = fy[:, 1:-1]

    # ensure $-255 \leq J \leq 255$
    jrng = np.max([np.max(np.abs(fx)), np.max(np.abs(fy))])
    assert jrng <= 255, "J must be in range [-255, 255]"

    ### 1609.01117 page 16

    hist, edges = np.histogramdd(
        np.vstack([fx.flatten(), fy.flatten()]).transpose(),
        bins=255,
    )

    ### 1609.01117 page 22

    deldensity = hist / hist.sum()
    deldensity = deldensity * -np.ma.log2(deldensity)
    entropy = np.sum(deldensity)
    entropy /= 2  # 4.3 Papoulis generalized sampling halves the delentropy

    log(f"entropy: {entropy}")
    log(f"entropy ratio: {entropy / 8.0}")

    # the reference image seems to be bitwise inverted, I don't know why.
    # the entropy doesn't change when inverted, so both are okay in
    # the previous computational steps.
    param_invert = True

    # gradimg = np.invert(grad) if param_invert else grad

    return (
        entropy,
        colourimg,
        greyimg,
        [
            # (gradimg, "Gradient", []),
            (deldensity, "Deldensity", ["hasbar"]),
        ],
    )


def delentropy2dv(args, colourimg, greyimg):
    ### 1609.01117 page 10

    grad = np.gradient(greyimg)
    fx = grad[0].astype(int)
    fy = grad[1].astype(int)

    # TODO: is this how fx and fy are combined?
    #       it's for plotting and not used in computation anyways,
    #       and it matches the image in the paper.
    grad = fx + fy

    # ensure $-255 \leq J \leq 255$
    jrng = np.max([np.max(np.abs(fx)), np.max(np.abs(fy))])
    assert jrng <= 255, "J must be in range [-255, 255]"

    ### 1609.01117 page 16

    hist, edgex, edgey = np.histogram2d(
        fx.flatten(),
        fy.flatten(),
        bins=255,
        range=[[-jrng, jrng], [-jrng, jrng]],
    )

    ### 1609.01117 page 22

    deldensity = hist / np.sum(hist)
    deldensity = deldensity * -np.ma.log2(deldensity)
    entropy = np.sum(deldensity)
    entropy /= 2  # 4.3 Papoulis generalized sampling halves the delentropy

    log(f"entropy: {entropy}")
    log(f"entropy ratio: {entropy / 8.0}")

    # the reference image seems to be bitwise inverted, I don't know why.
    # the entropy doesn't change when inverted, so both are okay in
    # the previous computational steps.
    param_invert = True

    gradimg = np.invert(grad) if param_invert else grad

    return (
        entropy,
        colourimg,
        greyimg,
        [
            (gradimg, "Gradient", ["hasbar"]),
            (deldensity, "Deldensity", ["hasbar"]),
        ],
    )


def delentropyndv(args, colourimg, greyimg):
    ### 1609.01117 page 10

    grad = [f.astype(int).flatten() for f in np.gradient(greyimg)]

    # ensure $-255 \leq J \leq 255$
    jrng = np.max(np.abs(grad))
    assert jrng <= 255, "J must be in range [-255, 255]"

    ### 1609.01117 page 16

    hist, edges = np.histogramdd(
        np.vstack(grad).transpose(),
        bins=255,
    )

    ### 1609.01117 page 22

    deldensity = hist / hist.sum()
    deldensity = deldensity * -np.ma.log2(deldensity)
    entropy = np.sum(deldensity)
    entropy /= 2  # 4.3 Papoulis generalized sampling halves the delentropy

    log(f"entropy: {entropy}")
    log(f"entropy ratio: {entropy / 8.0}")

    # the reference image seems to be bitwise inverted, I don't know why.
    # the entropy doesn't change when inverted, so both are okay in
    # the previous computational steps.
    param_invert = True

    # gradimg = np.invert(grad) if param_invert else grad

    return (
        entropy,
        colourimg,
        greyimg,
        [
            # (gradimg, "Gradient", []),
            (deldensity, "Deldensity", ["hasbar"]),
        ],
    )


def gradient2d(args, colourimg, greyimg):
    param_realgrad = True
    param_concave = True

    if param_realgrad:
        grads = np.gradient(greyimg)
        gradx = grads[0]
        grady = grads[1]
    else:
        gradx = cv.filter2D(
            greyimg,
            cv.CV_8U,
            cv.flip(np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]), -1),
            borderType=cv.BORDER_CONSTANT,
        )
        grady = cv.filter2D(
            greyimg,
            cv.CV_8U,
            cv.flip(np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]), -1),
            borderType=cv.BORDER_CONSTANT,
        )

    gradimg = (
        np.bitwise_or(gradx, grady)
        if not param_realgrad
        else (
            gradx + grady
            if not param_concave
            else np.invert(np.array(gradx + grady, dtype=int))
        )
    )

    log(f"gradient = {np.average(gradimg)} ± {np.std(gradimg)}")

    return (entropy, colourimg, greyimg, [(gradimg, "Gradient", [])])


def scikit2dr(args, colourimg, greyimg):
    # From scikit docs:
    # The entropy is computed using base 2 logarithm i.e. the filter returns
    # the minimum number of bits needed to encode the local gray level distribution.
    entimg = skentropy(greyimg.astype(np.uint8), skdisk(args.radius))
    entropy = entimg.mean()

    log(f"entropy: {entropy}")
    log(f"entropy ratio: {entropy / 8.0}")

    return (
        entropy,
        colourimg,
        greyimg,
        [(entimg, f"Scikit Entropy With Disk of Radius {args.radius}", ["hasbar"])],
    )


def shannon2dr(args, colourimg, greyimg):
    entimg = duplicate(greyimg)
    imgshape = entimg.shape

    kernsize = args.kernel_size
    kernrad = round((kernsize - 1) / 2)

    entropies = []
    for i in range(imgshape[0]):
        for j in range(imgshape[1]):
            region = greyimg[
                # ymax:ymin, xmax:xmin
                np.max([0, i - kernrad]) : np.min([imgshape[0], i + kernrad]),
                np.max([0, j - kernrad]) : np.min([imgshape[1], j + kernrad]),
            ].flatten()
            size = region.size

            probs = [np.size(region[region == i]) / size for i in set(region)]
            entropy = np.sum([p * np.log2(1 / p) for p in probs])

            entropies.append(entropy)
            entimg[i, j] = entropy

    # TODO: The average should not be used in latter computations.
    #       This is just to show.
    entropyavg = np.average(entropies)
    log(f"entropy = {entropyavg} ± {np.std(entropies)}")

    return (
        entropyavg,
        colourimg,
        greyimg,
        [
            (
                entimg,
                f"Entropy Map With {kernsize}x{kernsize} Kernel",
                ["hasbar"],
            )
        ],
    )


strtofunc = {
    "1d-kapur-variation": kapur1dv,
    "1d-shannon": shannon1d,
    "2d-delentropy": delentropy2d,
    "2d-delentropy-ndim": delentropynd,
    "2d-delentropy-variation": delentropy2dv,
    "2d-delentropy-variation-ndim": delentropyndv,
    "2d-gradient": gradient2d,
    "2d-regional-scikit": scikit2dr,
    "2d-regional-shannon": shannon2dr,
}
