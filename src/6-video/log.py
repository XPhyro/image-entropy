from sys import argv, stdout, stderr


execname = argv[0]

infoenabled = True
warnenabled = True
errenabled = True

spongeout = False
spongeerr = False
outcache = ""
errcache = ""


def _print(msg, file=stdout, end="\n"):
    global outcache, errcache

    assert file in (stdout, stderr)

    if file == stdout:
        if spongeout:
            outcache += msg + "\n"
        else:
            if outcache:
                print(outcache, end="")
                outcache = ""
            print(msg, file=file, end=end)
    elif spongeerr:
        errcache += msg + "\n"
    else:
        if errcache:
            print(errcache, end="")
            errcache = ""
        print(msg, file=file, end=end)


def dumpcaches():
    global spongeout, spongeerr

    spongeout = spongeerr = False
    _print("", file=stdout, end="")
    _print("", file=stderr, end="")


def info(*msgs):
    if not infoenabled:
        return
    for msg in msgs:
        _print(f"{execname}: {msg}")


def warn(*msgs):
    if not warnenabled:
        return
    for msg in msgs:
        _print(f"{execname}: {msg}", file=stderr)


def err(*msgs):
    if not errenabled:
        return
    for msg in msgs:
        _print(f"{execname}: {msg}", file=stderr)
