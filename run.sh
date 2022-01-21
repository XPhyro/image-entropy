#!/usr/bin/env sh

if command -v afgrep > /dev/null 2>&1; then
    filter() {
        afgrep -a 'main.py ['
    }
else
    printf "%s\n" "afgrep (from XPhyro/scripts) not found, falling back to grep." >&2
    filter() {
        grep '^main.py \['
    }
fi

unbuffer src/development/main.py -k 15 -M 0.995 -s 0.8 -S -m ./*.h5 data/* 2>&1 | filter
