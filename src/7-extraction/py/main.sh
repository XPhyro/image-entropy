#!/usr/bin/env sh

parseargs() {
    unset opttreatbinary optbuffersize optgreyscale optbinaryheight optbinarywidth optmaxframecount optpipath optbinarycolor optquiet
    while getopts "hsv:" OPT; do
        case "$OPT" in
            h)
                printf "%s" \
    "$0 [OPTION...] [FILE...] -- [ARG...]
       -B      treat given files as binary instead of video. if given, -H and -W are required.
       -h      Print help message and exit.
       -s      Print detailed status before initialisation.
       -v VPN  Set the VPN executable to use to VPN.
    "
                exit 0
                ;;
            s) optstatus=1;;
            v) VPN="$OPTARG";;
            *) printf "Invalid option given: %s\n" "$OPT"; exit 1;;
        esac
    done
    shift "$((OPTIND - 1))"
}
