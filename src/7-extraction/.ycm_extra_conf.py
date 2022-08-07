flags = [
    "-O3",
    "-DNDEBUG",
    "-std=c++20",  # YCM(or some other component) does not support c++23 yet and c++2b does not work "-Wall",
    "-Wall",
    "-Wextra",
    "-Werror",
    "-Wabi=11",
    "-Wno-unused-parameter",
    "-Wno-unused-result",
    "-Wno-implicit-fallthrough",
    "-Wno-sign-compare",
    "-Wstringop-overflow=4",
    "-Wfloat-equal",
    "-Wdouble-promotion",
    "-Wdisabled-optimization",
]

SOURCE_EXTENSIONS = [
    ".cpp",
    ".hpp",
]


def FlagsForFile(filename, **kwargs):
    return {"flags": flags, "do_cache": True}


def Settings(**kwargs):
    filename = kwargs.pop("filename")
    return FlagsForFile(filename, **kwargs)
