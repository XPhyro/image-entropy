#!/usr/bin/env sh

set -x
exec 2>&1

optfilter=1
unset pythonpath
while getopts "hp:s" OPT; do
    case "$OPT" in
        h)
            printf "%s" \
"usage: $0 [-h] [-s] [ARG [ARG ...]]

Quickly run the Python project using pre-set arguments.

   -h        show this help message and exit
   -p PATH   path to Python executable
   -s        do not filter output
"
            exit 0
            ;;
        p) pythonpath="$OPT";;
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

[ -n "$pythonpath" ] \
    || pythonpath="$(command -v python3.7 2>&1)" \
    || pythonpath="$(command -v python3 2>&1)" \
    || pythonpath="$(command -v python 2>&1)" \
    || {
        printf "Could not find a suitable executable for Python. Supply one with -p option.\n"
        exit 1
    }

if [ "$#" -ne 0 ]; then
    perf stat unbuffer "$pythonpath" src/development/main.py "$@"
else
    perf stat unbuffer "$pythonpath" src/development/main.py -k 15 -t 0.995 -s 0.8 -S \
        -m "$(find . -mindepth 1 -maxdepth 1 -type f \( -name "instance*.h5" -o -name "instance*.pkl" \) -print0 | head -n 1 -z)" \
        -M "$(find . -mindepth 1 -maxdepth 1 -type f -name "semantic*.h5" -print0 | head -n 1 -z)" \
        data/*
fi | filter
