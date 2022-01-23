import os


def makedirs(path):
    try:
        os.makedirs(path)
    except FileExistsError as err:
        if not os.path.isdir(path):
            raise err
