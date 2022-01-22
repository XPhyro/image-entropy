#!/usr/bin/env sh

optfilter=1
while getopts "hs" OPT; do
    case "$OPT" in
        h)
            printf "%s" \
"usage: $0 [-h] [-s] [ARG [ARG ...]]

Quickly run the Python project using pre-set arguments.

   -h      show this help message and exit
   -s      do not filter output
"
            exit 0
            ;;
        s) optfilter=0;;
        *) printf "Invalid option given: %s\n" "$OPT"; exit 1;;
    esac
done
shift "$((OPTIND - 1))"

if [ "$optfilter" -ne 0 ]; then
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
else
    filter() {
        cat
    }
fi

if [ "$#" -ne 0 ]; then
    unbuffer src/development/main.py "$@" 2>&1
else
    unbuffer src/development/main.py -k 15 -M 0.995 -s 0.8 -S -m ./*.h5 data/* 2>&1
fi | filter
