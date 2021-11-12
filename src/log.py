from sys import argv


execname = argv[0]


def log(*msgs):
    for msg in msgs:
        print(f"{execname}: {msg}")
