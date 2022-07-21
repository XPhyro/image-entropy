#!/usr/bin/env python3


import hashlib

import chacha


TESTKEY = [0x31] * 16
TESTIV = [0x41] * 8


def main():
    buf = [0x59] * 64
    cipher = chacha.ChaCha(TESTKEY, TESTIV)
    sha = hashlib.sha256()
    assert sha.digest_size < len(buf)  # see linux/drivers/char/random.c

    for _ in range(10):
        sha.update(bytes(cipher.next(buf)))
        rand = sha.hexdigest()
        print(rand)


if __name__ == "__main__":
    main()
