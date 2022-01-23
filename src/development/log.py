from datetime import datetime as dt
from os import devnull
from sys import argv, stdout, stderr


execname = argv[0][argv[0].rfind("/") + 1 :]


def _log(*args, file=devnull, **kwargs):
    print(f"{execname} [{dt.now()}]: ", file=file, end="")
    print(*args, file=file, sep="\n", **kwargs)


def loginfo(*args, **kwargs):
    _log(*args, file=stdout, **kwargs)


def logerr(*args, **kwargs):
    _log(*args, file=stderr, **kwargs)
