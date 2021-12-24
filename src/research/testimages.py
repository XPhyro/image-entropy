import numpy as np
from scipy.misc import face as spface, ascent as spascent


SEED = 3141592653


def gradlinearh(w=1024, h=1024):
    return np.outer(np.ones(256), np.arange(256)).astype(np.uint8)


def gradlinearv(w=1024, h=1024):
    return np.outer(np.arange(256), np.ones(256)).astype(np.uint8)


def gradconic(w=1024, h=1024):
    x, y = np.meshgrid(range(w), range(h))
    return (np.arctan2(h / 2 - y, w / 2 - x) * (255 / np.pi / 2)).astype(np.uint8)


def gradhalfconic(w=1024, h=1024):
    x, y = np.meshgrid(range(w), range(h))
    return (np.arctan2(h - y, w / 2 - x) * (255 / np.pi / 2)).astype(np.uint8)


def graddiag(w=1024, h=1024):
    x, y = np.meshgrid(range(w), range(h))
    return (np.arctan2(h - y, w - x) * (255 / np.pi)).astype(np.uint8)


def gradindex(w=1024, h=1024):
    return (np.indices((w, h)).sum(axis=0) % 256).astype(np.uint8)


def randunif(w=1024, h=1024):
    np.random.seed(SEED)
    return (np.random.rand(w, h) * 255).astype(np.uint8)


def pattern(w=1024, h=1024):
    x, y = np.meshgrid(range(w), range(h))
    return (np.cos(x) + np.sin(y)).astype(np.uint8)


def coscos(w=1024, h=1024):
    return (
        np.abs(np.cos(np.arange(w)[:, None]) * np.cos(np.arange(h)[None, :])) * 255
    ).astype(np.uint8)


def cos2cos2(w=1024, h=1024):
    return (
        np.abs(
            (np.cos(np.arange(w)[:, None]) ** 2) * (np.cos(np.arange(h)[None, :]) ** 2)
        )
        * 255
    ).astype(np.uint8)


def face(w=1024, h=1024):
    return spface(gray=True).astype(np.uint8)


def ascent(w=1024, h=1024):
    return spascent().astype(np.uint8)


def white(w=1024, h=1024):
    return (np.ones((w, h)) * 255).astype(np.uint8)


def black(w=1024, h=1024):
    return np.zeros((w, h)).astype(np.uint8)


def check(w=1024, h=1024):
    return ((np.indices((w, h)).sum(axis=0) % 2) * 255).astype(np.uint8)


strtofunc = {
    "gradlinearh": gradlinearh,
    "gradlinearv": gradlinearv,
    "gradconic": gradconic,
    "gradhalfconic": gradhalfconic,
    "graddiag": graddiag,
    "gradindex": gradindex,
    "randunif": randunif,
    "pattern": pattern,
    "coscos": coscos,
    "cos2cos2": cos2cos2,
    "face": face,
    "ascent": ascent,
    "white": white,
    "black": black,
    "check": check,
}
