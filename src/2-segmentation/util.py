import os
import sys

from log import logerr


def makedirs(path):
    try:
        os.makedirs(path)
    except FileExistsError as err:
        if not os.path.isdir(path):
            raise err


def basename(path):
    return path[path.rfind("/") + 1 :]


def _exit(code):
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(code)


def die(msg, code=1):
    logerr(msg)
    _exit(code)
