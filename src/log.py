from sys import argv


execname = argv[0]

infoenabled = True
warnenabled = True
errenabled = True


def info(*msgs):
    if not infoenabled:
        return

    for msg in msgs:
        print(f"{execname}: {msg}")


def warn(*msgs):
    if not warnenabled:
        return

    for msg in msgs:
        print(f"{execname}: {msg}")


def err(*msgs):
    if not errenabled:
        return

    for msg in msgs:
        print(f"{execname}: {msg}")
