from operator import itemgetter


def init(args):
    global np
    if args.gpu:
        import cupy as np
    else:
        import numpy as np


def log(args, arr):
    if args.gpu:
        arr[arr <= 0] = 1
        return np.log(arr)
    else:
        return np.ma.log(arr)


def log2(args, arr):
    if args.gpu:
        arr[arr <= 0] = 1
        return np.log2(arr)
    else:
        return np.ma.log2(arr)


def original(args, stack):
    ### 1609.01117 page 10

    # $\nabla f(n) \approx f(n) - f(n - 1)$
    fx = stack[:, 2:] - stack[:, :-2]
    fy = stack[2:, :] - stack[:-2, :]
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
    deldensity = deldensity * -(log if args.abstract_entropy else log2)(
        args, deldensity
    )
    entropy = np.sum(deldensity)
    entropy /= 2  # 4.3 Papoulis generalized sampling halves the delentropy

    return (entropy, (fx + fy, hist, deldensity))


def variationlight(args, stack):
    ### 1609.01117 page 10

    flatgrad = np.array(np.gradient(stack)).flatten()

    # ensure $-255 \leq J \leq 255$
    jrng = np.max(np.abs(flatgrad))
    assert jrng <= 255, "J must be in range [-255, 255]"

    ### 1609.01117 page 16

    hist, _ = np.histogram(
        flatgrad,
        bins=255,
    )

    ### 1609.01117 page 22

    deldensity = hist / hist.sum()
    deldensity = deldensity * -log2(args, deldensity)
    entropy = np.sum(deldensity)
    entropy /= 2  # 4.3 Papoulis generalized sampling halves the delentropy

    return (entropy, (flatgrad, hist, deldensity))


def variation(args, stack):
    ### 1609.01117 page 10

    grad = np.array([f.astype(int).flatten() for f in np.gradient(stack)])

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
    deldensity = deldensity * -log2(args, deldensity)
    entropy = np.sum(deldensity)
    entropy /= 2  # 4.3 Papoulis generalized sampling halves the delentropy

    return (entropy, (grad, hist, deldensity))


# TODO: generalise to n dimensions
def convolved(args, stack):
    kernshape = (args.kernel_size,) * 2
    kerns = np.lib.stride_tricks.as_strided(
        stack,
        tuple(np.subtract(stack.shape, kernshape) + 1) + kernshape,
        stack.strides * 2,
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
            deldensity = deldensity * -log2(args, deldensity)
            entropy = np.sum(deldensity)
            entropy /= 2
            kernent.append(entropy)

    kernent = np.reshape(kernent, itemgetter(0, 1)(kerns.shape))

    return (np.mean(kernent), (None, None, None))
