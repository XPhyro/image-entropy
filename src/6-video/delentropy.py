from operator import itemgetter

import numpy as np

import log


def original(args, colourimg, greyimg):
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
        bins=255,
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


def variation(args, colourimg, greyimg):
    ### 1609.01117 page 10

    grad = [f.astype(int).flatten() for f in np.gradient(greyimg)]

    # ensure $-255 \leq J \leq 255$
    jrng = np.max(np.abs(grad))
    assert jrng <= 255, "J must be in range [-255, 255]"

    ### 1609.01117 page 16

    hist, _ = np.histogramdd(
        np.vstack(grad).transpose(),
        bins=255,
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


# TODO: generalise to n dimensions
def convolved(args, colourimg, greyimg):
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
                bins=255,
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
