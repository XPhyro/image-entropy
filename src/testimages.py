import numpy as np


def gradlinearh(w=1024, h=1024):
    return np.outer(np.ones(256), np.arange(256)).astype(np.uint8)


def gradlinearv(w=1024, h=1024):
    return np.outer(np.arange(256), np.ones(256)).astype(np.uint8)


strtofunc = {
    "horizontal-linear-gradient": gradlinearh,
    "vertical-linear-gradient": gradlinearv,
}
