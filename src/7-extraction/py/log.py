from sys import argv, stdout, stderr


prefix = argv[0]

noout = False

infoenabled = True
warnenabled = True
errenabled = True

spongeout = False
spongeerr = False

outcache = ""
errcache = ""


def init(extractorid):
    global prefix

    if extractorid is not None:
        prefix = f"{argv[0]}.{extractorid}"


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
    if not noout:
        _print("", file=stdout, end="")
    _print("", file=stderr, end="")


def raw(bits):
    return stdout.buffer.write(bits)


def info(*msgs):
    if noout:
        warn(*msgs)
        return

    if not infoenabled:
        return
    for msg in msgs:
        _print(f"{prefix}: {msg}", file=stdout)


def warn(*msgs):
    if not warnenabled:
        return
    for msg in msgs:
        _print(f"{prefix}: {msg}", file=stderr)


def err(*msgs):
    if not errenabled:
        return
    for msg in msgs:
        _print(f"{prefix}: {msg}", file=stderr)
