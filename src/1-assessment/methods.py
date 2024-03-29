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
from operator import itemgetter

from scipy import stats
import numpy as np

import log


def init(args):
    match args.method:
        case "2d-gradient-cnn":
            from scipy.ndimage.filters import gaussian_filter
        case "2d-regional-scikit":
            from skimage.filters.rank import entropy as skentropy
            from skimage.morphology import disk as skdisk


def kapur1dv(args, colourimg, greyimg):
    hist = np.histogram(greyimg, bins=256, range=(0, 256))[0]
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

    log.info(
        f"entropy: {entropy}",
        f"threshold: {threshold}",
        f"entropy ratio: {entropy / 8.0}",
    )

    entimg = np.where(greyimg < threshold, greyimg, 0)

    return (
        f"{entropy} after 1 iteration",
        colourimg,
        greyimg,
        [(entimg, "Kapur Threshold", ["hasbar"])],
    )


def shannon1d(args, colourimg, greyimg):
    _, counts = np.unique(greyimg.flatten(), return_counts=True)
    entropy = stats.entropy(counts, base=2)

    log.info(
        f"entropy: {entropy}",
        f"entropy ratio: {entropy / 8.0}",
    )

    return (entropy, None, None, None)


def delentropy2d(args, colourimg, greyimg):
    ### 1609.01117 page 10

    # $\nabla f(n) \approx f(n) - f(n - 1)$
    fx = greyimg[:, 2:] - greyimg[:, :-2]
    fy = greyimg[2:, :] - greyimg[:-2, :]
    # fix shape
    fx = fx[1:-1, :]
    fy = fy[:, 1:-1]

    grad = fx + fy

    # ensure $-255 \leq J \leq 255$
    jrng = np.max([np.max(np.abs(fx)), np.max(np.abs(fy))])
    assert jrng <= 255, "J must be in range [-255, 255]"

    ### 1609.01117 page 16, eq 17

    hist, _, _ = np.histogram2d(
        fx.flatten(),
        fy.flatten(),
        bins=256,
    )

    ### 1609.01117 page 20, eq 22

    deldensity = hist / np.sum(hist)
    deldensity = deldensity * -np.ma.log2(deldensity)
    entropy = np.sum(deldensity)

    entropy /= 2  # 4.3 Papoulis generalized sampling halves the delentropy

    log.info(
        f"entropy: {entropy}",
        f"entropy ratio: {entropy / 8.0}",
    )

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

    hist, _ = np.histogramdd(
        np.vstack([fx.flatten(), fy.flatten()]).transpose(),
        bins=256,
    )

    ### 1609.01117 page 22

    deldensity = hist / hist.sum()
    deldensity = deldensity * -np.ma.log2(deldensity)
    entropy = np.sum(deldensity)
    entropy /= 2  # 4.3 Papoulis generalized sampling halves the delentropy

    log.info(
        f"entropy: {entropy}",
        f"entropy ratio: {entropy / 8.0}",
    )

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

    grad = fx + fy

    # ensure $-255 \leq J \leq 255$
    jrng = np.max([np.max(np.abs(fx)), np.max(np.abs(fy))])
    assert jrng <= 255, "J must be in range [-255, 255]"

    ### 1609.01117 page 16

    hist, _, _ = np.histogram2d(
        fx.flatten(),
        fy.flatten(),
        bins=256,
        range=[[-jrng, jrng], [-jrng, jrng]],
    )

    ### 1609.01117 page 22

    deldensity = hist / np.sum(hist)
    deldensity = deldensity * -np.ma.log2(deldensity)
    entropy = np.sum(deldensity)
    entropy /= 2  # 4.3 Papoulis generalized sampling halves the delentropy

    log.info(
        f"entropy: {entropy}",
        f"entropy ratio: {entropy / 8.0}",
    )

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


def delentropy2dv2(args, colourimg, greyimg):
    ### 1609.01117 page 10

    grad = np.gradient(greyimg).astype(int)
    firstgrad = grad[0] + grad[1]

    grad = np.gradient(firstgrad)
    secondgrad = grad[0] + grad[1]

    fx = grad[0].astype(int)
    fy = grad[1].astype(int)
    grad = fx + fy

    # ensure $-255 \leq J \leq 255$
    jrng = np.max([np.max(np.abs(fx)), np.max(np.abs(fy))])
    assert jrng <= 255, "J must be in range [-255, 255]"

    ### 1609.01117 page 16

    hist, _, _ = np.histogram2d(
        fx.flatten(),
        fy.flatten(),
        bins=256,
        range=[[-jrng, jrng], [-jrng, jrng]],
    )

    ### 1609.01117 page 22

    deldensity = hist / np.sum(hist)
    deldensity = deldensity * -np.ma.log2(deldensity)
    entropy = np.sum(deldensity)
    entropy /= 2  # 4.3 Papoulis generalized sampling halves the delentropy

    log.info(
        f"entropy: {entropy}",
        f"entropy ratio: {entropy / 8.0}",
    )

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
            (firstgrad[0] + firstgrad[1], "First Degree Gradient", ["hasbar"]),
            (secondgrad[0] + secondgrad[1], "Second Degree Gradient", ["hasbar"]),
            (deldensity, "Deldensity", ["hasbar"]),
        ],
    )


def gradient2dc(args, colourimg, greyimg):
    ### 1609.01117 page 10

    grad = np.gradient(greyimg)
    fx = grad[0].astype(int)
    fy = grad[1].astype(int)

    grad = fx + fy

    # ensure $-255 \leq J \leq 255$
    jrng = np.max([np.max(np.abs(fx)), np.max(np.abs(fy))])
    assert jrng <= 255, "J must be in range [-255, 255]"

    ### 1609.01117 page 16

    hist, _, _ = np.histogram2d(
        fx.flatten(),
        fy.flatten(),
        bins=256,
        range=[[-jrng, jrng], [-jrng, jrng]],
    )

    ### 1609.01117 page 22

    deldensity = hist / np.sum(hist)
    deldensity = deldensity * -np.ma.log2(deldensity)
    halfdeldensity = (
        deldensity / 2
    )  # 4.3 Papoulis generalized sampling halves the delentropy

    kernshape = (args.kernel_size,) * 2
    kerndensity = np.einsum(
        "ijkl->ij",
        np.lib.stride_tricks.as_strided(
            deldensity,
            tuple(np.subtract(deldensity.shape, kernshape) + 1) + kernshape,
            deldensity.strides * 2,
        ),
    )
    kerngrad = np.einsum(
        "ijkl->ij",
        np.lib.stride_tricks.as_strided(
            grad,
            tuple(np.subtract(grad.shape, kernshape) + 1) + kernshape,
            grad.strides * 2,
        ),
    )

    mu = 0.99
    roigrad = np.abs(kerngrad)
    roigradflat = roigrad.flatten()
    mean = np.mean(roigrad)
    roigradbound = (
        mean,
        *stats.t.interval(
            mu, len(roigradflat) - 1, loc=mean, scale=stats.sem(roigradflat)
        ),
    )
    roigrad = roigrad.astype(float)
    roigrad[roigrad < roigradbound[2]] = 0
    roigrad /= np.linalg.norm(roigrad)

    sigma = 7
    roigradblurred = gaussian_filter(roigrad, sigma=sigma)

    entropy = np.sum(halfdeldensity)

    log.info(
        f"entropy: {entropy}",
        f"entropy ratio: {entropy / 8.0}",
    )

    return (
        entropy,
        colourimg,
        greyimg,
        [
            (grad, "Gradient", ["hasbar", "forcecolour"]),
            (kerngrad, "Convolved Gradient", ["hasbar", "forcecolour"]),
            (deldensity, "Deldensity", ["hasbar", "forcecolour"]),
            (kerndensity, "Convolved Deldensity", ["hasbar", "forcecolour"]),
            (roigrad, "Regions of Interest", ["hasbar", "forcecolour"]),
            (roigradblurred, "Blurred Regions of Interest", ["hasbar", "forcecolour"]),
        ],
    )


def delentropy2dvc(args, colourimg, greyimg):
    kernshape = (args.kernel_size,) * 2
    kerns = np.lib.stride_tricks.as_strided(
        greyimg,
        tuple(np.subtract(greyimg.shape, kernshape) + 1) + kernshape,
        greyimg.strides * 2,
    )
    kernent = []

    for i in kerns:
        for _ in i:
            fx = grad[0].astype(int)
            fy = grad[1].astype(int)

            grad = fx + fy

            jrng = np.max([np.max(np.abs(fx)), np.max(np.abs(fy))])
            assert jrng <= 255, "J must be in range [-255, 255]"

            hist, _, _ = np.histogram2d(
                fx.flatten(),
                fy.flatten(),
                bins=256,
                range=[[-jrng, jrng], [-jrng, jrng]],
            )

            deldensity = hist / np.sum(hist)
            deldensity = deldensity * -np.ma.log2(deldensity)
            entropy = np.sum(deldensity)
            entropy /= 2
            kernent.append(entropy)

    kernent = np.reshape(kernent, itemgetter(0, 1)(kerns.shape))

    return (
        np.mean(kernent),
        colourimg,
        greyimg,
        [
            (kernent, "Kernelised Delentropy", ["hasbar", "forcecolour"]),
        ],
    )


def delentropyndv(args, colourimg, greyimg):
    ### 1609.01117 page 10

    grad = [f.astype(int).flatten() for f in np.gradient(greyimg)]

    # ensure $-255 \leq J \leq 255$
    jrng = np.max(np.abs(grad))
    assert jrng <= 255, "J must be in range [-255, 255]"

    ### 1609.01117 page 16

    hist, _ = np.histogramdd(
        np.vstack(grad).transpose(),
        bins=256,
    )

    ### 1609.01117 page 22

    deldensity = hist / hist.sum()
    deldensity = deldensity * -np.ma.log2(deldensity)
    entropy = np.sum(deldensity)
    entropy /= 2  # 4.3 Papoulis generalized sampling halves the delentropy

    log.info(
        f"entropy: {entropy}",
        f"entropy ratio: {entropy / 8.0}",
    )

    return (
        entropy,
        colourimg,
        greyimg,
        [
            (deldensity, "Deldensity", ["hasbar"]),
        ],
    )


def scikit2dr(args, colourimg, greyimg):
    # From scikit docs:
    # The entropy is computed using base 2 logarithm i.e. the filter returns
    # the minimum number of bits needed to encode the local gray level distribution.
    entimg = skentropy(greyimg.astype(np.uint8), skdisk(args.radius))
    entropy = entimg.mean()

    log.info(
        f"entropy: {entropy}",
        f"entropy ratio: {entropy / 8.0}",
    )

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

    # the average should not be used in latter computations, it's just for printing
    entropyavg = np.average(entropies)
    log.info(
        f"entropy = {entropyavg} ± {np.std(entropies)}",
    )

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


def shannon2dv1(args, colourimg, greyimg):
    h, w = greyimg.shape
    fy, fx = [], []

    for y in range(h):
        _, counts = np.unique(greyimg[y, :].flatten(), return_counts=True)
        entropy = stats.entropy(counts, base=2)
        fy.append(entropy)
    for x in range(w):
        _, counts = np.unique(greyimg[:, x].flatten(), return_counts=True)
        entropy = stats.entropy(counts, base=2)
        fx.append(entropy)

    (histy, _), (histx, _) = (
        np.histogram(fy, bins=args.bins_count, range=[0, 8]),
        np.histogram(fx, bins=args.bins_count, range=[0, 8]),
    )
    hist = [histy, histx]

    entdensity = hist / np.sum(hist)
    entdensity = entdensity * -np.ma.log2(entdensity)
    entropy = np.sum(entdensity)

    log.info(
        f"entropy: {entropy}",
        f"entropy ratio: {entropy / 8.0}",
    )

    return (
        entropy,
        colourimg,
        greyimg,
        [],
    )


def shannon2dv2(args, colourimg, greyimg):
    h, w = greyimg.shape

    if h != w:
        log.err("image must be a square, skipping.")
        return (
            -1,
            colourimg,
            greyimg,
            [],
        )

    fy, fx = [], []

    for y in range(h):
        _, counts = np.unique(greyimg[y, :].flatten(), return_counts=True)
        entropy = stats.entropy(counts, base=2)
        fy.append(entropy)
    for x in range(w):
        _, counts = np.unique(greyimg[:, x].flatten(), return_counts=True)
        entropy = stats.entropy(counts, base=2)
        fx.append(entropy)

    hist, _, _ = np.histogram2d(fx, fy, bins=args.bins_count, range=[[0, 8], [0, 8]])

    entdensity = hist / np.sum(hist)
    entdensity = entdensity * -np.ma.log2(entdensity)
    entropy = np.sum(entdensity)

    log.info(
        f"entropy: {entropy}",
        f"entropy ratio: {entropy / 8.0}",
    )

    return (
        entropy,
        colourimg,
        greyimg,
        [],
    )


def shannon2dv3(args, colourimg, greyimg):
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

    hist, _ = np.histogram(entimg.flatten(), bins=args.bins_count, range=[0, 8])

    entdensity = hist / np.sum(hist)
    entdensity = entdensity * -np.ma.log2(entdensity)
    entropy = np.sum(entdensity)

    log.info(
        f"entropy = {entropy}",
    )

    return (
        entropy,
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


def shannon2dv4(args, colourimg, greyimg):
    kernshape = (args.kernel_size,) * 2
    kerns = np.lib.stride_tricks.as_strided(
        greyimg,
        tuple(np.subtract(greyimg.shape, kernshape) + 1) + kernshape,
        greyimg.strides * 2,
    )
    entropies = []

    for kern in kerns:
        size = kern.size

        probs = [np.size(kern[kern == i]) / size for i in set(kern)]
        entropy = np.sum([p * np.log2(1 / p) for p in probs])

        entropies.append(entropy)

    hist, _ = np.histogram(entropies, bins=args.bins_count, range=[0, 8])

    entdensity = hist / np.sum(hist)
    entdensity = entdensity * -np.ma.log2(entdensity)
    entropy = np.sum(entdensity)

    return (
        entropy,
        colourimg,
        greyimg,
        [],
    )


def shannon2d(args, colourimg, greyimg):
    img = np.array(greyimg).flatten()
    hist, _, _ = np.histogram2d(img, img, bins=256)

    entdensity = hist / np.sum(hist)
    entdensity = entdensity * -np.ma.log2(entdensity)
    entropy = np.sum(entdensity)

    log.info(f"entropy = {entropy}")

    return (
        entropy,
        colourimg,
        greyimg,
        [],
    )


strtofunc = {
    "1d-kapur-variation": kapur1dv,
    "1d-shannon": shannon1d,
    "2d-delentropy": delentropy2d,
    "2d-delentropy-ndim": delentropynd,
    "2d-delentropy-variation": delentropy2dv,
    "2d-delentropy-variation-cnn": delentropy2dvc,
    "2d-delentropy-variation-ndim": delentropyndv,
    "2d-delentropy-variation-second-degree": delentropy2dv2,
    "2d-gradient-cnn": gradient2dc,
    "2d-regional-scikit": scikit2dr,
    "2d-regional-shannon": shannon2dr,
    "2d-shannon-variation-1": shannon2dv1,
    "2d-shannon-variation-2": shannon2dv2,
    "2d-shannon-variation-3": shannon2dv3,
    "2d-shannon-variation-4": shannon2dv4,
    "2d-shannon": shannon2d,
}
