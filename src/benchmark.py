import time

import numpy as np

from . import log


try:
    from pypapi import events, papi_high as high

    haspapi = True
except:
    haspapi = False


def checkevent(event):
    try:
        high.start_counters([event])
        high.stop_counters()
        return True
    except:
        return False


def measurepapi(event, count, func, funcargs):
    if not checkevent(event):
        return -1

    measure = 0

    for i in range(count):
        high.start_counters([event])
        func(*funcargs)
        measure += high.stop_counters()[0] / count

    return measure


def benchmark(args, func, funcargs):
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

        if checkevent(events.PAPI_FP_OPS):
            fp = measurepapi(events.PAPI_FP_OPS, count, func, funcargs)
        elif checkevent(events.PAPI_SP_OPS) and checkevent(events.PAPI_DP_OPS):
            log.warn(
                "PAPI_FP_OPS unavailable. "
                + "FLO/FLOPS will be estimated using PAPI_SP_OPS and PAPI_DP_OPS."
            )

            fp = measurepapi(events.PAPI_DP_OPS, count, func, funcargs) + measurepapi(
                events.PAPI_SP_OPS, count, func, funcargs
            )
        else:
            fp = -1

        vp = max(
            -1,
            measurepapi(events.PAPI_VEC_DP, count, func, funcargs)
            + measurepapi(events.PAPI_VEC_SP, count, func, funcargs),
        )
        cyc = measurepapi(events.PAPI_REF_CYC, count, func, funcargs)

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
