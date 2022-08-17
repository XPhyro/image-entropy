from log import err as logerr


np = None  # suppress Pylint E0602


def init(args):
    global np
    if args.gpu:
        try:
            import cupy as np
        except ModuleNotFoundError:
            logerr("cupy not found, falling back to numpy")
            import numpy as np
    else:
        import numpy as np


def _log(args, arr):
    if not args.gpu:
        return np.ma.log(arr)
    arr[arr <= 0] = 1
    return np.log(arr)


def _log2(args, arr):
    if not args.gpu:
        return np.ma.log2(arr)
    arr[arr <= 0] = 1
    return np.log2(arr)


def _autolog(args, arr):
    return _log(args, arr) if args.abstract_entropy else _log2(args, arr)


def variationlight(args, stack):
    stackarr = np.array(stack)
    stackshape = stackarr.shape
    ### 1609.01117 page 10

    flatgrad = np.array(np.gradient(stackarr)).astype(np.int16).flatten()
    del stackarr

    # # ensure $-255 \leq J \leq 255$
    # jrng = np.max(np.abs(flatgrad))
    # assert jrng <= 255, "J must be in range [-255, 255]"

    ### 1609.01117 page 16

    hist, _ = np.histogram(
        flatgrad,
        bins=256,
    )
    del flatgrad

    ### 1609.01117 page 22

    deldensity = hist / hist.sum()
    del hist
    deldensity = deldensity * -_autolog(args, deldensity)
    entropy = np.sum(deldensity)
    del deldensity
    entropy /= len(
        stackshape
    )  # 4.3 Papoulis generalized sampling halves the delentropy

    return entropy


def variation(args, stack):
    stackarr = np.array(stack)
    stackshape = stackarr.shape
    ### 1609.01117 page 10

    grad = np.array([f.astype(np.int16).flatten() for f in np.gradient(stackarr)])
    del stackarr

    # # ensure $-255 \leq J \leq 255$
    # jrng = np.max(np.abs(grad))
    # assert jrng <= 255, "J must be in range [-255, 255]"

    ### 1609.01117 page 16

    hist, _ = np.histogramdd(
        np.vstack(grad).transpose(),
        bins=256,
    )
    del grad

    ### 1609.01117 page 22

    deldensity = hist / hist.sum()
    del hist
    deldensity = deldensity * -_autolog(args, deldensity)
    entropy = np.sum(deldensity)
    del deldensity
    entropy /= len(
        stackshape
    )  # 4.3 Papoulis generalized sampling halves the delentropy

    return entropy
