#!/usr/bin/env python3
# References:
#   [1]: linux/drivers/char/random.c


import hashlib

import chacha
from constants import TESTKEY, TESTIV


def main():
    buf = [0x59] * 64
    cipher = chacha.ChaCha(TESTKEY, TESTIV)

    # hash to not expose internal states, see [1]
    sha = hashlib.sha256()
    assert sha.digest_size < len(buf)  # see [1]

    for _ in range(10):
        sha.update(bytes(cipher.next(buf)))
        rand = sha.hexdigest()
        print(rand)


if __name__ == "__main__":
    main()
