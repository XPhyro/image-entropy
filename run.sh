#!/usr/bin/env sh

set -x
exec 2>&1

logerrq() {
    printf "%s\n" "$@" >&2
    exit 1
}

optfilter=1
unset pythonpath optmodel
while getopts "ahP:ps" OPT; do
    case "$OPT" in
        a) optmodel="ade20k";;
        h)
            printf "%s" \
"usage: $0 [-a] [-h] [-m] [-P] [-p] [-s] [ARG [ARG ...]] -- [ARG [ARG ...]]

Quickly run the Python project using pre-set arguments.

   -a        enable Ade20k semantic segmentation
   -h        show this help message and exit
   -P PATH   path to Python executable
   -p        enable Pascalvoc semantic segmentation
   -s        do not filter output
"
            exit 0
            ;;
        P) pythonpath="$OPT";;
        p) optmodel="pascalvoc";;
        s) optfilter=0;;
        *) printf "Invalid option given: %s\n" "$OPT"; exit 1;;
    esac
done
shift "$((OPTIND - 1))"

[ -z "$optmodel" ] && logerrq "One of -a, -m or -p must be given.\n"

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
    || logerrq "Could not find a suitable executable for Python. Supply one with -p option.\n"

if [ "$#" -ne 0 ]; then
    perf stat unbuffer \
        "$pythonpath" src/development/main.py "$@"
else
    case "$optmodel" in
        ade20k)
            opt="-a"
            model="$(
                find . -mindepth 1 -maxdepth 1 -type f -name "semantic*ade20k*.h5" -print0 \
                    | head -n 1 -z
            )"
            datadir="data/color"
            ;;
        pascalvoc)
            opt="-p"
            model="$(
                find . -mindepth 1 -maxdepth 1 -type f -name "semantic*pascalvoc*.h5" -print0 \
                    | head -n 1 -z
            )"
            datadir="data/color"
            ;;
    esac
    perf stat unbuffer \
        "$pythonpath" src/development/main.py \
            -k 15 -t 0.995 -s 0.8 -S "$opt" "$model" "$datadir"/*
fi | filter
