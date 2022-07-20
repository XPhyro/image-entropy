#!/usr/bin/env python3

import chacha

TESTKEY = [0x31] * 16
TESTIV = [0x41] * 8


def main():
    buf = [0x59] * 64
    cipher = chacha.ChaCha(TESTKEY, TESTIV)

    for _ in range(10):
        randarr = cipher.next(buf)
        rand = 0
        for i, r in enumerate(randarr):
            rand += 2**i * r

        print(rand)


if __name__ == "__main__":
    main()
