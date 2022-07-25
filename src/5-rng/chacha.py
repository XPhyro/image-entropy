#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =======================================================================
#
# chacha.py
# ---------
# Simple model of the ChaCha stream cipher. Used as a reference for
# the HW implementation. The code follows the structure of the
# HW implementation as much as possible.
#
#
# Copyright (c) 2013 Secworks Sweden AB
# Copyright (c) 2022 Berke Kocaoğlu
# Original Author: Joachim Strömbergson
#
# Redistribution and use in source and binary forms, with or
# without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# =======================================================================


# -------------------------------------------------------------------
# Python module imports.
# -------------------------------------------------------------------
import sys
from constants import TAU, SIGMA


# ---------------------------------------------------------------
# _b2w()
#
# Given a list of four bytes returns the little endian
# 32 bit word representation of the bytes.
# ---------------------------------------------------------------
def _b2w(u32):
    return (u32[0] + (u32[1] << 8) + (u32[2] << 16) + (u32[3] << 24)) & 0xFFFFFFFF


# ---------------------------------------------------------------
# _w2b()
#
# Given a 32-bit word returns a list of set of four bytes
# that is the little endian byte representation of the word.
# ---------------------------------------------------------------
def _w2b(u32):
    return [
        (u32 & 0x000000FF),
        ((u32 & 0x0000FF00) >> 8),
        ((u32 & 0x00FF0000) >> 16),
        ((u32 & 0xFF000000) >> 24),
    ]


# -------------------------------------------------------------------
# ChaCha()
# -------------------------------------------------------------------
class ChaCha:
    """
    Initialise with set_key_iv(...) and use next(...) to rotate.
    """

    # ---------------------------------------------------------------
    # __init__()
    #
    # Given the key, iv initializes the state of the cipher.
    # The number of rounds used can be set. By default 8 rounds
    # are used. Accepts a list of either 16 or 32 bytes as key.
    # Accepts a list of 8 bytes as IV.
    # ---------------------------------------------------------------
    def __init__(self, key, iv, rounds=8, verbose=0):
        self.state = [0] * 16
        self.x = [0] * 16
        self.rounds = rounds
        self.verbose = verbose
        self.set_key_iv(key, iv)

    # ---------------------------------------------------------------
    # set_key_iv()
    #
    # Set key and iv. Basically reinitialize the cipher.
    # This also resets the block counter.
    # ---------------------------------------------------------------
    def set_key_iv(self, key, iv):
        keyword0 = self._b2w(key[0:4])
        keyword1 = self._b2w(key[4:8])
        keyword2 = self._b2w(key[8:12])
        keyword3 = self._b2w(key[12:16])

        if len(key) == 16:
            self.state[0] = TAU[0]
            self.state[1] = TAU[1]
            self.state[2] = TAU[2]
            self.state[3] = TAU[3]
            self.state[4] = keyword0
            self.state[5] = keyword1
            self.state[6] = keyword2
            self.state[7] = keyword3
            self.state[8] = keyword0
            self.state[9] = keyword1
            self.state[10] = keyword2
            self.state[11] = keyword3

        elif len(key) == 32:
            keyword4 = self._b2w(key[16:20])
            keyword5 = self._b2w(key[20:24])
            keyword6 = self._b2w(key[24:28])
            keyword7 = self._b2w(key[28:32])
            self.state[0] = SIGMA[0]
            self.state[1] = SIGMA[1]
            self.state[2] = SIGMA[2]
            self.state[3] = SIGMA[3]
            self.state[4] = keyword0
            self.state[5] = keyword1
            self.state[6] = keyword2
            self.state[7] = keyword3
            self.state[8] = keyword4
            self.state[9] = keyword5
            self.state[10] = keyword6
            self.state[11] = keyword7
        else:
            print(f"Key length of {len(key) * 8} bits, is not supported.")

        # Common state init for both key lengths.
        self.block_counter = [0, 0]
        self.state[12] = self.block_counter[0]
        self.state[13] = self.block_counter[1]
        self.state[14] = self._b2w(iv[0:4])
        self.state[15] = self._b2w(iv[4:8])

        if self.verbose:
            print("State after init:")
            self._print_state()

    # ---------------------------------------------------------------
    # next()
    #
    # Encyp/decrypt the next block. This also updates the
    # internal state and increases the block counter.
    # ---------------------------------------------------------------
    def next(self, data_in):
        # Copy the current internal state to the temporary state x.
        self.x = self.state[:]

        if self.verbose:
            print("State before round processing.")
            self._print_state()

        if self.verbose:
            print("X before round processing:")
            self._print_x()

        # Update the internal state by performing
        # (rounds / 2) double rounds.
        for i in range(int(self.rounds / 2)):
            if self.verbose > 1:
                print(f"Doubleround 0x{i:02x}:")
            self._doubleround()
            if self.verbose > 1:
                print("")

        if self.verbose:
            print("X after round processing:")
            self._print_x()

        # Update the internal state by adding the elements
        # of the temporary state to the internal state.
        self.state = [((self.state[i] + self.x[i]) & 0xFFFFFFFF) for i in range(16)]

        if self.verbose:
            print("State after round processing.")
            self._print_state()

        bytestate = []
        for i in self.state:
            bytestate += self._w2b(i)

        # Create the data out words.
        data_out = [data_in[i] ^ bytestate[i] for i in range(64)]

        # Update the block counter.
        self._inc_counter()

        return data_out

    # ---------------------------------------------------------------
    # _doubleround()
    #
    # Perform the two complete rounds that comprises the
    # double round.
    # ---------------------------------------------------------------
    def _doubleround(self):
        if self.verbose > 0:
            print("Start of double round processing.")

        self._quarterround(0, 4, 8, 12)
        if self.verbose > 1:
            print("X after QR 0")
            self._print_x()
        self._quarterround(1, 5, 9, 13)
        if self.verbose > 1:
            print("X after QR 1")
            self._print_x()
        self._quarterround(2, 6, 10, 14)
        if self.verbose > 1:
            print("X after QR 2")
            self._print_x()
        self._quarterround(3, 7, 11, 15)
        if self.verbose > 1:
            print("X after QR 3")
            self._print_x()

        self._quarterround(0, 5, 10, 15)
        if self.verbose > 1:
            print("X after QR 4")
            self._print_x()
        self._quarterround(1, 6, 11, 12)
        if self.verbose > 1:
            print("X after QR 5")
            self._print_x()
        self._quarterround(2, 7, 8, 13)
        if self.verbose > 1:
            print("X after QR 6")
            self._print_x()
        self._quarterround(3, 4, 9, 14)
        if self.verbose > 1:
            print("X after QR 7")
            self._print_x()

        if self.verbose > 0:
            print("End of double round processing.")

    # ---------------------------------------------------------------
    #  _quarterround()
    #
    # Updates four elements in the state vector x given by
    # their indices.
    # ---------------------------------------------------------------
    def _quarterround(self, ai, bi, ci, di):
        # Extract four elemenst from x using the qi tuple.
        a, b, c, d = self.x[ai], self.x[bi], self.x[ci], self.x[di]

        if self.verbose > 1:
            print("Indata to quarterround:")
            print("X state indices:", ai, bi, ci, di)
            print(f"a = 0x{a:08x}, b = 0x{b:08x}, c = 0x{c:08x}, d = 0x{d:08x}")
            print("")

        a0 = (a + b) & 0xFFFFFFFF
        d0 = d ^ a0
        d1 = ((d0 << 16) + (d0 >> 16)) & 0xFFFFFFFF
        c0 = (c + d1) & 0xFFFFFFFF
        b0 = b ^ c0
        b1 = ((b0 << 12) + (b0 >> 20)) & 0xFFFFFFFF
        a1 = (a0 + b1) & 0xFFFFFFFF
        d2 = d1 ^ a1
        d3 = ((d2 << 8) + (d2 >> 24)) & 0xFFFFFFFF
        c1 = (c0 + d3) & 0xFFFFFFFF
        b2 = b1 ^ c1
        b3 = ((b2 << 7) + (b2 >> 25)) & 0xFFFFFFFF

        if self.verbose > 2:
            print("Intermediate values:")
            print(f"a0 = 0x{a0:08x}, a1 = 0x{a1:08x}")
            print(
                f"b0 = 0x{b0:08x}, b1 = 0x{b1:08x}, b2 = 0x{b2:08x}, b3 = 0x{b3:08x}"
                % (b0, b1, b2, b3)
            )
            print(f"c0 = 0x{c0:08x}, c1 = 0x{c1:08x}")
            print(
                f"d0 = 0x{d0:08x}, d1 = 0x{d1:08x}, d2 = 0x{d2:08x}, d3 = 0x{d3:08x}"
                % (d0, d1, d2, d3)
            )
            print("")

        a_prim = a1
        b_prim = b3
        c_prim = c1
        d_prim = d3

        if self.verbose > 1:
            print("Outdata from quarterround:")
            print(
                f"a_prim = 0x{a_prim:08x}, b_prim = 0x{b_prim:08x}, c_prim = 0x{c_prim:08x}, d_prim = 0x{d_prim:08x}"
            )
            print("")

        # Update the four elemenst in x using the qi tuple.
        self.x[ai], self.x[bi] = a_prim, b_prim
        self.x[ci], self.x[di] = c_prim, d_prim

    # ---------------------------------------------------------------
    # _inc_counter()
    #
    # Increase the 64 bit block counter.
    # ---------------------------------------------------------------
    def _inc_counter(self):
        self.block_counter[0] += 1 & 0xFFFFFFFF
        if not self.block_counter[0] % 0xFFFFFFFF:
            self.block_counter[1] += 1 & 0xFFFFFFFF

    # ---------------------------------------------------------------
    # _print_state()
    #
    # Print the internal state.
    # ---------------------------------------------------------------
    def _print_state(self):
        print(
            f" 0: 0x{self.state[0]:08x},  1: 0x{self.state[1]:08x},  2: 0x{self.state[2]:08x},  3: 0x{self.state[3]:08x}"
        )
        print(
            f" 4: 0x{self.state[0]:08x},  5: 0x{self.state[1]:08x},  6: 0x{self.state[2]:08x},  7: 0x{self.state[3]:08x}"
        )
        print(
            f" 8: 0x{self.state[0]:08x},  9: 0x{self.state[1]:08x}, 10: 0x{self.state[2]:08x}, 11: 0x{self.state[3]:08x}"
        )
        print(
            f"12: 0x{self.state[0]:08x}, 13: 0x{self.state[1]:08x}, 14: 0x{self.state[2]:08x}, 15: 0x{self.state[3]:08x}"
        )
        print("")

    # ---------------------------------------------------------------
    # _print_x()
    #
    # Print the temporary state X.
    # ---------------------------------------------------------------
    def _print_x(self):
        print(
            f" 0: 0x{self.x[0]:08x},  1: 0x{self.x[1]:08x},  2: 0x{self.x[2]:08x},  3: 0x{self.x[3]:08x}"
        )
        print(
            f" 4: 0x{self.x[5]:08x},  5: 0x{self.x[5]:08x},  6: 0x{self.x[6]:08x},  7: 0x{self.x[7]:08x}"
        )
        print(
            f" 8: 0x{self.x[8]:08x},  9: 0x{self.x[9]:08x}, 10: 0x{self.x[10]:08x}, 11: 0x{self.x[11]:08x}"
        )
        print(
            f"12: 0x{self.x[12]:08x}, 13: 0x{self.x[13]:08x}, 14: 0x{self.x[14]:08x}, 15: 0x{self.x[15]:08x}"
        )
        print("")


# -------------------------------------------------------------------
# print_block()
#
# Print a given block (list) of bytes ordered in
# rows of eight bytes.
# -------------------------------------------------------------------
def print_block(block):
    for i in range(0, len(block), 8):
        print(
            f"0x{block[i + 0]:02x} 0x{block[i + 1]:02x} 0x{block[i + 2]:02x} 0x{block[i + 3]:02x} 0x{block[i + 4]:02x} 0x{block[i + 5]:02x} 0x{block[i + 6]:02x} 0x{block[i + 7]:02x}"
        )


# -------------------------------------------------------------------
# check_block()
#
# Compare the result block with the expected block and print()
# if the result for the given test case was correct or not.
# -------------------------------------------------------------------
def check_block(result, expected, test_case):
    if result == expected:
        print(f"SUCCESS: {test_case} was correct.")
    else:
        print(f"ERROR: {test_case} was not correct.")
        print("Expected:")
        print_block(expected)
        print("")
        print("Result:")
        print_block(result)
    print("")


# -------------------------------------------------------------------
# main()
#
# If executed tests the ChaCha class using known test vectors.
# -------------------------------------------------------------------
def main():
    print("Testing the ChaCha Python model.")
    print("--------------------------------")
    print()

    # Testing with TC1-128-8.
    # All zero inputs. IV all zero. 128 bit key, 8 rounds.
    print("TC1-128-8: All zero inputs. 128 bit key, 8 rounds.")
    key1 = [0x00] * 16
    iv1 = [0x00] * 8
    expected1 = [
        0xE2,
        0x8A,
        0x5F,
        0xA4,
        0xA6,
        0x7F,
        0x8C,
        0x5D,
        0xEF,
        0xED,
        0x3E,
        0x6F,
        0xB7,
        0x30,
        0x34,
        0x86,
        0xAA,
        0x84,
        0x27,
        0xD3,
        0x14,
        0x19,
        0xA7,
        0x29,
        0x57,
        0x2D,
        0x77,
        0x79,
        0x53,
        0x49,
        0x11,
        0x20,
        0xB6,
        0x4A,
        0xB8,
        0xE7,
        0x2B,
        0x8D,
        0xEB,
        0x85,
        0xCD,
        0x6A,
        0xEA,
        0x7C,
        0xB6,
        0x08,
        0x9A,
        0x10,
        0x18,
        0x24,
        0xBE,
        0xEB,
        0x08,
        0x81,
        0x4A,
        0x42,
        0x8A,
        0xAB,
        0x1F,
        0xA2,
        0xC8,
        0x16,
        0x08,
        0x1B,
    ]
    block1 = [0x00] * 64
    cipher1 = ChaCha(key1, iv1, verbose=0)
    result1 = cipher1.next(block1)
    check_block(result1, expected1, "TC1-128-8")
    print()

    # Testing with TC1-128-12.
    # All zero inputs. IV all zero. 128 bit key, 12 rounds.
    print("TC1-128-12: All zero inputs. 128 bit key, 12 rounds.")
    key1 = [0x00] * 16
    iv1 = [0x00] * 8
    expected1 = [
        0xE1,
        0x04,
        0x7B,
        0xA9,
        0x47,
        0x6B,
        0xF8,
        0xFF,
        0x31,
        0x2C,
        0x01,
        0xB4,
        0x34,
        0x5A,
        0x7D,
        0x8C,
        0xA5,
        0x79,
        0x2B,
        0x0A,
        0xD4,
        0x67,
        0x31,
        0x3F,
        0x1D,
        0xC4,
        0x12,
        0xB5,
        0xFD,
        0xCE,
        0x32,
        0x41,
        0x0D,
        0xEA,
        0x8B,
        0x68,
        0xBD,
        0x77,
        0x4C,
        0x36,
        0xA9,
        0x20,
        0xF0,
        0x92,
        0xA0,
        0x4D,
        0x3F,
        0x95,
        0x27,
        0x4F,
        0xBE,
        0xFF,
        0x97,
        0xBC,
        0x84,
        0x91,
        0xFC,
        0xEF,
        0x37,
        0xF8,
        0x59,
        0x70,
        0xB4,
        0x50,
    ]
    block1 = [0x00] * 64
    cipher1 = ChaCha(key1, iv1, rounds=12, verbose=0)
    result1 = cipher1.next(block1)
    check_block(result1, expected1, "TC1-128-12")
    print()

    # Testing with TC1-128-20.
    # All zero inputs. IV all zero. 128 bit key, 20 rounds.
    print("TC1-128-20: All zero inputs. 128 bit key, 20 rounds.")
    key1 = [0x00] * 16
    iv1 = [0x00] * 8
    expected1 = [
        0x89,
        0x67,
        0x09,
        0x52,
        0x60,
        0x83,
        0x64,
        0xFD,
        0x00,
        0xB2,
        0xF9,
        0x09,
        0x36,
        0xF0,
        0x31,
        0xC8,
        0xE7,
        0x56,
        0xE1,
        0x5D,
        0xBA,
        0x04,
        0xB8,
        0x49,
        0x3D,
        0x00,
        0x42,
        0x92,
        0x59,
        0xB2,
        0x0F,
        0x46,
        0xCC,
        0x04,
        0xF1,
        0x11,
        0x24,
        0x6B,
        0x6C,
        0x2C,
        0xE0,
        0x66,
        0xBE,
        0x3B,
        0xFB,
        0x32,
        0xD9,
        0xAA,
        0x0F,
        0xDD,
        0xFB,
        0xC1,
        0x21,
        0x23,
        0xD4,
        0xB9,
        0xE4,
        0x4F,
        0x34,
        0xDC,
        0xA0,
        0x5A,
        0x10,
        0x3F,
    ]
    block1 = [0x00] * 64
    cipher1 = ChaCha(key1, iv1, rounds=20, verbose=0)
    result1 = cipher1.next(block1)
    check_block(result1, expected1, "TC1-128-20")
    print()

    # Testing with TC1-256-8.
    # All zero inputs. IV all zero. 256 bit key, 8 rounds.
    print("TC1-256-8: All zero inputs. 256 bit key, 8 rounds.")
    key1 = [0x00] * 32
    iv1 = [0x00] * 8
    expected1 = [
        0x3E,
        0x00,
        0xEF,
        0x2F,
        0x89,
        0x5F,
        0x40,
        0xD6,
        0x7F,
        0x5B,
        0xB8,
        0xE8,
        0x1F,
        0x09,
        0xA5,
        0xA1,
        0x2C,
        0x84,
        0x0E,
        0xC3,
        0xCE,
        0x9A,
        0x7F,
        0x3B,
        0x18,
        0x1B,
        0xE1,
        0x88,
        0xEF,
        0x71,
        0x1A,
        0x1E,
        0x98,
        0x4C,
        0xE1,
        0x72,
        0xB9,
        0x21,
        0x6F,
        0x41,
        0x9F,
        0x44,
        0x53,
        0x67,
        0x45,
        0x6D,
        0x56,
        0x19,
        0x31,
        0x4A,
        0x42,
        0xA3,
        0xDA,
        0x86,
        0xB0,
        0x01,
        0x38,
        0x7B,
        0xFD,
        0xB8,
        0x0E,
        0x0C,
        0xFE,
        0x42,
    ]
    block1 = [0x00] * 64
    cipher1 = ChaCha(key1, iv1, verbose=0)
    result1 = cipher1.next(block1)
    check_block(result1, expected1, "TC1-256-8")
    print()

    # Testing with TC1-256-12.
    # All zero inputs. IV all zero. 256 bit key, 12 rounds.
    print("TC1-256-12: All zero inputs. 256 bit key, 12 rounds.")
    key1 = [0x00] * 32
    iv1 = [0x00] * 8
    expected1 = [
        0x9B,
        0xF4,
        0x9A,
        0x6A,
        0x07,
        0x55,
        0xF9,
        0x53,
        0x81,
        0x1F,
        0xCE,
        0x12,
        0x5F,
        0x26,
        0x83,
        0xD5,
        0x04,
        0x29,
        0xC3,
        0xBB,
        0x49,
        0xE0,
        0x74,
        0x14,
        0x7E,
        0x00,
        0x89,
        0xA5,
        0x2E,
        0xAE,
        0x15,
        0x5F,
        0x05,
        0x64,
        0xF8,
        0x79,
        0xD2,
        0x7A,
        0xE3,
        0xC0,
        0x2C,
        0xE8,
        0x28,
        0x34,
        0xAC,
        0xFA,
        0x8C,
        0x79,
        0x3A,
        0x62,
        0x9F,
        0x2C,
        0xA0,
        0xDE,
        0x69,
        0x19,
        0x61,
        0x0B,
        0xE8,
        0x2F,
        0x41,
        0x13,
        0x26,
        0xBE,
    ]
    block1 = [0x00] * 64
    cipher1 = ChaCha(key1, iv1, rounds=12, verbose=0)
    result1 = cipher1.next(block1)
    check_block(result1, expected1, "TC1-256-12")
    print()

    # Testing with TC1-256-20.
    # All zero inputs. IV all zero. 256 bit key, 20 rounds.
    print("TC1-256-20: All zero inputs. 256 bit key, 20 rounds.")
    key1 = [0x00] * 32
    iv1 = [0x00] * 8
    expected1 = [
        0x76,
        0xB8,
        0xE0,
        0xAD,
        0xA0,
        0xF1,
        0x3D,
        0x90,
        0x40,
        0x5D,
        0x6A,
        0xE5,
        0x53,
        0x86,
        0xBD,
        0x28,
        0xBD,
        0xD2,
        0x19,
        0xB8,
        0xA0,
        0x8D,
        0xED,
        0x1A,
        0xA8,
        0x36,
        0xEF,
        0xCC,
        0x8B,
        0x77,
        0x0D,
        0xC7,
        0xDA,
        0x41,
        0x59,
        0x7C,
        0x51,
        0x57,
        0x48,
        0x8D,
        0x77,
        0x24,
        0xE0,
        0x3F,
        0xB8,
        0xD8,
        0x4A,
        0x37,
        0x6A,
        0x43,
        0xB8,
        0xF4,
        0x15,
        0x18,
        0xA1,
        0x1C,
        0xC3,
        0x87,
        0xB6,
        0x69,
        0xB2,
        0xEE,
        0x65,
        0x86,
    ]
    block1 = [0x00] * 64
    cipher1 = ChaCha(key1, iv1, rounds=20, verbose=0)
    result1 = cipher1.next(block1)
    check_block(result1, expected1, "TC1-256-20")
    print()

    # Testing with TC2-128-8.
    # Single bit set in key. IV all zero. 128 bit key.
    print("TC2-128-8: One bit in key set. IV all zeros. 128 bit key, 8 rounds.")
    key2 = [
        0x01,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
    ]
    iv2 = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
    expected2 = [
        0x03,
        0xA7,
        0x66,
        0x98,
        0x88,
        0x60,
        0x5A,
        0x07,
        0x65,
        0xE8,
        0x35,
        0x74,
        0x75,
        0xE5,
        0x86,
        0x73,
        0xF9,
        0x4F,
        0xC8,
        0x16,
        0x1D,
        0xA7,
        0x6C,
        0x2A,
        0x3A,
        0xA2,
        0xF3,
        0xCA,
        0xF9,
        0xFE,
        0x54,
        0x49,
        0xE0,
        0xFC,
        0xF3,
        0x8E,
        0xB8,
        0x82,
        0x65,
        0x6A,
        0xF8,
        0x3D,
        0x43,
        0x0D,
        0x41,
        0x09,
        0x27,
        0xD5,
        0x5C,
        0x97,
        0x2A,
        0xC4,
        0xC9,
        0x2A,
        0xB9,
        0xDA,
        0x37,
        0x13,
        0xE1,
        0x9F,
        0x76,
        0x1E,
        0xAA,
        0x14,
    ]
    block2 = [0x00] * 64
    cipher2 = ChaCha(key2, iv2, verbose=0)
    result2 = cipher2.next(block2)
    check_block(result2, expected2, "TC2-128-8")
    print()

    # Testing with TC2-256-8.
    # Single bit set in key. IV all zero. 256 bit key.
    print("TC2-256-8: One bit in key set. IV all zeros. 256 bit key, 8 rounds.")
    key2 = [
        0x01,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
    ]
    iv2 = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
    expected2 = [
        0xCF,
        0x5E,
        0xE9,
        0xA0,
        0x49,
        0x4A,
        0xA9,
        0x61,
        0x3E,
        0x05,
        0xD5,
        0xED,
        0x72,
        0x5B,
        0x80,
        0x4B,
        0x12,
        0xF4,
        0xA4,
        0x65,
        0xEE,
        0x63,
        0x5A,
        0xCC,
        0x3A,
        0x31,
        0x1D,
        0xE8,
        0x74,
        0x04,
        0x89,
        0xEA,
        0x28,
        0x9D,
        0x04,
        0xF4,
        0x3C,
        0x75,
        0x18,
        0xDB,
        0x56,
        0xEB,
        0x44,
        0x33,
        0xE4,
        0x98,
        0xA1,
        0x23,
        0x8C,
        0xD8,
        0x46,
        0x4D,
        0x37,
        0x63,
        0xDD,
        0xBB,
        0x92,
        0x22,
        0xEE,
        0x3B,
        0xD8,
        0xFA,
        0xE3,
        0xC8,
    ]
    block2 = [0x00] * 64
    cipher2 = ChaCha(key2, iv2, verbose=0)
    result2 = cipher2.next(block2)
    check_block(result2, expected2, "TC2-256-8")
    print()

    # Testing with TC3-128-8.
    # All zero key. Single bit in IV set. 128 bit key.
    print("TC3-128-8: All zero key. Single bit in IV set. 128 bit key, 8 rounds.")
    key3 = [
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
    ]
    iv3 = [0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]

    expected3 = [
        0x25,
        0xF5,
        0xBE,
        0xC6,
        0x68,
        0x39,
        0x16,
        0xFF,
        0x44,
        0xBC,
        0xCD,
        0x12,
        0xD1,
        0x02,
        0xE6,
        0x92,
        0x17,
        0x66,
        0x63,
        0xF4,
        0xCA,
        0xC5,
        0x3E,
        0x71,
        0x95,
        0x09,
        0xCA,
        0x74,
        0xB6,
        0xB2,
        0xEE,
        0xC8,
        0x5D,
        0xA4,
        0x23,
        0x6F,
        0xB2,
        0x99,
        0x02,
        0x01,
        0x2A,
        0xDC,
        0x8F,
        0x0D,
        0x86,
        0xC8,
        0x18,
        0x7D,
        0x25,
        0xCD,
        0x1C,
        0x48,
        0x69,
        0x66,
        0x93,
        0x0D,
        0x02,
        0x04,
        0xC4,
        0xEE,
        0x88,
        0xA6,
        0xAB,
        0x35,
    ]
    block3 = [0x00] * 64
    cipher3 = ChaCha(key3, iv3, verbose=0)
    result3 = cipher3.next(block3)
    check_block(result3, expected3, "TC3-128-8")
    print()

    # Testing with TC4-128-8.
    # All bits in key IV are set. 128 bit key, 8 rounds.
    print("TC4-128-8: All bits in key IV are set. 128 bit key, 8 rounds.")
    key4 = [
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
    ]
    iv4 = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]
    expected4 = [
        0x22,
        0x04,
        0xD5,
        0xB8,
        0x1C,
        0xE6,
        0x62,
        0x19,
        0x3E,
        0x00,
        0x96,
        0x60,
        0x34,
        0xF9,
        0x13,
        0x02,
        0xF1,
        0x4A,
        0x3F,
        0xB0,
        0x47,
        0xF5,
        0x8B,
        0x6E,
        0x6E,
        0xF0,
        0xD7,
        0x21,
        0x13,
        0x23,
        0x04,
        0x16,
        0x3E,
        0x0F,
        0xB6,
        0x40,
        0xD7,
        0x6F,
        0xF9,
        0xC3,
        0xB9,
        0xCD,
        0x99,
        0x99,
        0x6E,
        0x6E,
        0x38,
        0xFA,
        0xD1,
        0x3F,
        0x0E,
        0x31,
        0xC8,
        0x22,
        0x44,
        0xD3,
        0x3A,
        0xBB,
        0xC1,
        0xB1,
        0x1E,
        0x8B,
        0xF1,
        0x2D,
    ]
    block4 = [0x00] * 64
    cipher4 = ChaCha(key4, iv4, verbose=0)
    result4 = cipher4.next(block4)
    check_block(result4, expected4, "TC4-128-8")
    print()

    # Testing with TC5-128-8
    print("TC5-128-8: Even bits set. 128 bit key, 8 rounds.")
    key5 = [0x55] * 16
    iv5 = [0x55] * 8
    expected5 = [
        0xF0,
        0xA2,
        0x3B,
        0xC3,
        0x62,
        0x70,
        0xE1,
        0x8E,
        0xD0,
        0x69,
        0x1D,
        0xC3,
        0x84,
        0x37,
        0x4B,
        0x9B,
        0x2C,
        0x5C,
        0xB6,
        0x01,
        0x10,
        0xA0,
        0x3F,
        0x56,
        0xFA,
        0x48,
        0xA9,
        0xFB,
        0xBA,
        0xD9,
        0x61,
        0xAA,
        0x6B,
        0xAB,
        0x4D,
        0x89,
        0x2E,
        0x96,
        0x26,
        0x1B,
        0x6F,
        0x1A,
        0x09,
        0x19,
        0x51,
        0x4A,
        0xE5,
        0x6F,
        0x86,
        0xE0,
        0x66,
        0xE1,
        0x7C,
        0x71,
        0xA4,
        0x17,
        0x6A,
        0xC6,
        0x84,
        0xAF,
        0x1C,
        0x93,
        0x19,
        0x96,
    ]
    block5 = [0x00] * 64
    cipher5 = ChaCha(key5, iv5, verbose=0)
    result5 = cipher5.next(block5)
    check_block(result5, expected5, "TC5-128-8")
    print()

    # Testing with TC6-128-8
    print("TC6-128-8: Odd bits set. 128 bit key, 8 rounds.")
    key6 = [0xAA] * 16
    iv6 = [0xAA] * 8
    expected6 = [
        0x31,
        0x2D,
        0x95,
        0xC0,
        0xBC,
        0x38,
        0xEF,
        0xF4,
        0x94,
        0x2D,
        0xB2,
        0xD5,
        0x0B,
        0xDC,
        0x50,
        0x0A,
        0x30,
        0x64,
        0x1E,
        0xF7,
        0x13,
        0x2D,
        0xB1,
        0xA8,
        0xAE,
        0x83,
        0x8B,
        0x3B,
        0xEA,
        0x3A,
        0x7A,
        0xB0,
        0x38,
        0x15,
        0xD7,
        0xA4,
        0xCC,
        0x09,
        0xDB,
        0xF5,
        0x88,
        0x2A,
        0x34,
        0x33,
        0xD7,
        0x43,
        0xAC,
        0xED,
        0x48,
        0x13,
        0x6E,
        0xBA,
        0xB7,
        0x32,
        0x99,
        0x50,
        0x68,
        0x55,
        0xC0,
        0xF5,
        0x43,
        0x7A,
        0x36,
        0xC6,
    ]
    block6 = [0x00] * 64
    cipher6 = ChaCha(key6, iv6, verbose=0)
    result6 = cipher6.next(block6)
    check_block(result6, expected6, "TC6-128-8")
    print()

    # Testing with TC7-128-8
    print(
        "TC7-128-8: Key and IV are increasing, decreasing patterns. 128 bit key, 8 rounds."
    )
    key7 = [
        0x00,
        0x11,
        0x22,
        0x33,
        0x44,
        0x55,
        0x66,
        0x77,
        0x88,
        0x99,
        0xAA,
        0xBB,
        0xCC,
        0xDD,
        0xEE,
        0xFF,
    ]
    iv7 = [0x0F, 0x1E, 0x2D, 0x3C, 0x4B, 0x59, 0x68, 0x77]
    expected7 = [
        0xA7,
        0xA6,
        0xC8,
        0x1B,
        0xD8,
        0xAC,
        0x10,
        0x6E,
        0x8F,
        0x3A,
        0x46,
        0xA1,
        0xBC,
        0x8E,
        0xC7,
        0x02,
        0xE9,
        0x5D,
        0x18,
        0xC7,
        0xE0,
        0xF4,
        0x24,
        0x51,
        0x9A,
        0xEA,
        0xFB,
        0x54,
        0x47,
        0x1D,
        0x83,
        0xA2,
        0xBF,
        0x88,
        0x88,
        0x61,
        0x58,
        0x6B,
        0x73,
        0xD2,
        0x28,
        0xEA,
        0xAF,
        0x82,
        0xF9,
        0x66,
        0x5A,
        0x5A,
        0x15,
        0x5E,
        0x86,
        0x7F,
        0x93,
        0x73,
        0x1B,
        0xFB,
        0xE2,
        0x4F,
        0xAB,
        0x49,
        0x55,
        0x90,
        0xB2,
        0x31,
    ]
    block7 = [0x00] * 64
    cipher7 = ChaCha(key7, iv7, verbose=2)
    result7 = cipher7.next(block7)
    check_block(result7, expected7, "TC7-128-8")
    print()

    # Testing with TC8-128-8
    print("TC8-128-8: Random inputs. 128 bit key, 8 rounds.")
    key8 = [
        0xC4,
        0x6E,
        0xC1,
        0xB1,
        0x8C,
        0xE8,
        0xA8,
        0x78,
        0x72,
        0x5A,
        0x37,
        0xE7,
        0x80,
        0xDF,
        0xB7,
        0x35,
    ]
    iv8 = [0x1A, 0xDA, 0x31, 0xD5, 0xCF, 0x68, 0x82, 0x21]
    expected8 = [
        0x6A,
        0x87,
        0x01,
        0x08,
        0x85,
        0x9F,
        0x67,
        0x91,
        0x18,
        0xF3,
        0xE2,
        0x05,
        0xE2,
        0xA5,
        0x6A,
        0x68,
        0x26,
        0xEF,
        0x5A,
        0x60,
        0xA4,
        0x10,
        0x2A,
        0xC8,
        0xD4,
        0x77,
        0x00,
        0x59,
        0xFC,
        0xB7,
        0xC7,
        0xBA,
        0xE0,
        0x2F,
        0x5C,
        0xE0,
        0x04,
        0xA6,
        0xBF,
        0xBB,
        0xEA,
        0x53,
        0x01,
        0x4D,
        0xD8,
        0x21,
        0x07,
        0xC0,
        0xAA,
        0x1C,
        0x7C,
        0xE1,
        0x1B,
        0x7D,
        0x78,
        0xF2,
        0xD5,
        0x0B,
        0xD3,
        0x60,
        0x2B,
        0xBD,
        0x25,
        0x94,
    ]
    block8 = [0x00] * 64
    cipher8 = ChaCha(key8, iv8, verbose=0)
    result8 = cipher8.next(block8)
    check_block(result8, expected8, "TC8-128-8")
    print()


# -------------------------------------------------------------------
# __name__
# Python thingy which allows the file to be run standalone as
# well as parsed from within a Python interpreter.
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Run the main function.
    sys.exit(main())

# =======================================================================
# EOF chacha.py
# =======================================================================
