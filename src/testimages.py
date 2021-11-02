import numpy as np


SEED = 3141592653


def gradlinearh(w=1024, h=1024):
    return np.outer(np.ones(256), np.arange(256)).astype(np.uint8)


def gradlinearv(w=1024, h=1024):
    return np.outer(np.arange(256), np.ones(256)).astype(np.uint8)


def gradconic(w=1024, h=1024):
    x, y = np.meshgrid(range(w), range(h))
    return (np.arctan2(h / 2 - y, w / 2 - x) * (255 / np.pi / 2)).astype(np.uint8)


def randunif(w=1024, h=1024):
    np.random.seed(SEED)
    return (np.random.rand(w, h) * 255).astype(np.uint8)


def patterncossin(w=1024, h=1024):
    x, y = np.meshgrid(range(w), range(h))
    return (np.cos(x) + np.sin(y)).astype(np.uint8)


strtofunc = {
    "horizontal-linear-gradient": gradlinearh,
    "vertical-linear-gradient": gradlinearv,
    "conic-gradient": gradconic,
    "uniform-random": randunif,
    "cos-sin-pattern": patterncossin,
}
