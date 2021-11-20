import time

import numpy as np

from . import log


try:
    from pypapi import events, papi_high as high

    haspapi = True
except:
    haspapi = False


def _checkevent(event):
    """
    Check for the existence of PAPI events.

    This function checks for the existence of the given PAPI event by starting and
    stopping a counter on it. Unless you know what you are doing, do not use this
    directly; instead, use ``benchmark``.

    Parameters
    ----------
    event : int
        Event to check.

    Returns
    -------
    bool
        Whether the event exists.

    See Also
    --------
    benchmark : TODO

    Examples
    --------
    >>> _checkevent(events.PAPI_FP_OPS)
    False
    >>> _checkevent(events.PAPI_SP_OPS)
    True
    """
    try:
        high.start_counters([event])
        high.stop_counters()
        return True
    except:
        return False


def _measurepapi(event, count, func, funcargs):
    """
    Measure the given event for the given function and arguments for count times.

    Unless you know what you are doing, do not use this directly; instead, use ``benchmark``.

    Parameters
    ----------
    event : int
        Event to measure.
    count : int
        Number of times the measurement should be done.
    func : function
        Function to measure on.
    funcargs : tuple
        Arguments to pass to the function.

    Returns
    -------
    int
        Mean of measured event count.

    See Also
    --------
    benchmark : TODO

    Examples
    --------
    >>> _measurepapi(events.PAPI_FP_OPS, 10, methods.method, (args, colourimg, greyimg))
    234567
    >>> _measurepapi(events.PAPI_DP_OPS, 10, methods.method, (args, colourimg, greyimg))
    123456
    """
    if not _checkevent(event):
        return -1

    measure = 0

    for i in range(count):
        high.start_counters([event])
        func(*funcargs)
        measure += high.stop_counters()[0]

    return measure // count


def benchmark(args, func, funcargs):
    """
    Benchmark the given function and arguments using various events.

    Benchmark the given functions, passing the given arguments to it, using various
    events. The events used are: PAPI_FP_OPS, PAPI_SP_OPS, PAPI_DP_OPS, PAPI_VEC_sP, PAPI_VEC_SP, PAPI_REF_CYC.

    Parameters
    ----------
    args : Namespace
        The return value of argparse.ArgumentParser(...).parse_args(...).
    func : function
        Function to benchmark.
    funcargs : tuple
        Arguments to pass to the function.

    Returns
    -------
    None

    Examples
    --------
    >>> benchmark(args, methods.method, (args, colourimg, greyimg))
    FLO................: 234567
    VO.................: 123456
    FLOPS..............: 23456789
    VOPS...............: 12345678
    FLOPC..............: 0.912356
    VOPC...............: 0.812345
    process time.......: 1.123456
    processor cycles...: 12345678
    """
    if not args.print_performance:
        return

    if not haspapi:
        log.warn(
            "performance statistics requested but pypapi is not available. "
            + "only process time will be benchmarked."
        )

        log.infoenabled = False
        timebeg = time.process_time()
        func(*funcargs)
        timeend = time.process_time()
        log.infoenabled = True
    else:
        log.infoenabled = False

        count = args.performance_count

        if _checkevent(events.PAPI_FP_OPS):
            fp = _measurepapi(events.PAPI_FP_OPS, count, func, funcargs)
        elif _checkevent(events.PAPI_SP_OPS) and _checkevent(events.PAPI_DP_OPS):
            log.warn(
                "PAPI_FP_OPS unavailable. "
                + "FLO/FLOPS will be estimated using PAPI_SP_OPS and PAPI_DP_OPS."
            )

            fp = _measurepapi(events.PAPI_DP_OPS, count, func, funcargs) + _measurepapi(
                events.PAPI_SP_OPS, count, func, funcargs
            )
        else:
            fp = -1

        vp = max(
            -1,
            _measurepapi(events.PAPI_VEC_DP, count, func, funcargs)
            + _measurepapi(events.PAPI_VEC_SP, count, func, funcargs),
        )
        cyc = _measurepapi(events.PAPI_REF_CYC, count, func, funcargs)

        cputime = time.process_time()
        func(*funcargs)
        cputime = time.process_time() - cputime

        log.infoenabled = True

        log.info(
            f"FLO................: {fp}",
            f"VO.................: {vp}",
            f"FLOPS..............: {fp / cputime}",
            f"VOPS...............: {vp / cputime}",
            f"FLOPC..............: {fp / cyc}",
            f"VOPC...............: {vp / cyc}",
            f"process time.......: {cputime}",
            f"processor cycles...: {cyc}",
        )
