import os


def makedirs(path):
    try:
        os.makedirs(path)
    except FileExistsError as err:
        if not os.path.isdir(path):
            raise err


def basename(path):
    return path[path.rfind("/") + 1 :]
