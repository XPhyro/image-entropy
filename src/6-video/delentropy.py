from operator import itemgetter


def init(args):
    global np
    if args.gpu:
        import cupy as np
    else:
        import numpy as np


def log(args, arr):
    if not args.gpu:
        return np.ma.log(arr)
    arr[arr <= 0] = 1
    return np.log(arr)


def log2(args, arr):
    if not args.gpu:
        return np.ma.log2(arr)
    arr[arr <= 0] = 1
    return np.log2(arr)


def autolog(args, arr):
    return log(args, arr) if args.abstract_entropy else log2(args, arr)


def variationlight(args, stack):
    stack = np.array(stack)
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
    deldensity = deldensity * -autolog(args, deldensity)
    entropy = np.sum(deldensity)
    entropy /= 2 ** (
        len(stack.shape) - 1
    )  # 4.3 Papoulis generalized sampling halves the delentropy

    return (entropy, (flatgrad, hist, deldensity))


def variation(args, stack):
    stack = np.array(stack)
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
    deldensity = deldensity * -autolog(args, deldensity)
    entropy = np.sum(deldensity)
    entropy /= 2 ** (
        len(stack.shape) - 1
    )  # 4.3 Papoulis generalized sampling halves the delentropy

    return (entropy, (grad, hist, deldensity))
