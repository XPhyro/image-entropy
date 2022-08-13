#!/usr/bin/env sh

set -x
exec 2>&1

loginfo() {
    printf "$execname: %s\n" "$@"
}

logerrq() {
    printf "$execname: %s\n" "$@" >&2
    exit 1
}

setfilter() {
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
}

segmentation() {
    optfilter=1
    unset pythonpath optmodel
    while getopts "ahP:ps" OPT; do
        case "$OPT" in
            a) optmodel="ade20k";;
            h)
                printf "%s" \
"usage: $0 [-a] [-h] [-m] [-P] [-p] [-s] [ARG [ARG ...]] -- [ARG [ARG ...]]

Quickly run the segmentation sub-project using pre-set arguments.

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
            *) printf "invalid option given: %s\n" "$OPT"; exit 1;;
        esac
    done
    shift "$((OPTIND - 1))"

    setfilter

    [ -n "$pythonpath" ] \
        || pythonpath="$(command -v python3.7 2> /dev/null)" \
        || pythonpath="$(command -v python3 2> /dev/null)" \
        || pythonpath="$(command -v python 2> /dev/null)" \
        || logerrq "could not find a suitable executable for Python, supply one with -p option\n"

    if [ "$#" -ne 0 ]; then
        perf stat \
            unbuffer \
                "$pythonpath" src/2-segmentation/main.py "$@"
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
        perf stat \
            unbuffer \
                "$pythonpath" src/2-segmentation/main.py \
                    -k 15 -t 0.995 -s 0.8 -S "$opt" "$model" "$datadir"/*
    fi | filter
}

video() {
    optfilter=1
    while getopts "ahP:ps" OPT; do
        case "$OPT" in
            a) optmodel="ade20k";;
            h)
                printf "%s" \
"usage: $0 [-a] [-h] [-m] [-P] [-p] [-s] [ARG [ARG ...]] -- [ARG [ARG ...]]

Quickly run the video sub-project using pre-set arguments.

   -h        show this help message and exit
   -s        do not filter output
"
                exit 0
                ;;
            s) optfilter=0;;
            *) printf "invalid option given: %s\n" "$OPT"; exit 1;;
        esac
    done
    shift "$((OPTIND - 1))"

    setfilter

    if [ "$#" -ne 0 ]; then
        perf stat \
            unbuffer \
                src/6-video/main.py "$@"
    else
        loginfo "presets are not yet implemented" # TODO: presets not yet implemented
    fi | filter
}

shannon() {
    src/1-assessment/main.py -m 1d-shannon -n -- "$@" | grep -E ' (processing file|entropy): '
}

rawshannon() {
    if [ "$#" -eq 0 ]; then
        ent < /dev/stdin
    else
        for i; do
            ent "$i"
        done
    fi
}

execname="$0"

[ "$#" -eq 0 ] && logerrq "no subcommand given"
subcmd="$1"
shift
case "$subcmd" in
    2|s|seg|segmentation|2-segmentation) segmentation "$@";;
    6|v|vid|video|6-video) video "$@";;
    shannon) shannon "$@";;
    raw-shannon) rawshannon "$@";;
    *) logerrq "invalid or not-supported subcommand";;
esac

