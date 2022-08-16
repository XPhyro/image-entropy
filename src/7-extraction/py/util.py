import time


def gettime():
    return time.clock_gettime(time.CLOCK_MONOTONIC)


def savestart():
    savestart.starttime = gettime()


def getdiff():
    return gettime() - savestart.starttime
