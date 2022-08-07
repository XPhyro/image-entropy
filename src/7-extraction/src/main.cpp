#include <chrono>
#include <iostream>
#include <numbers>
#include <ratio>
#include <sstream>
#include <string>
#include <thread>

#include <err.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <cstdutil.hpp>
#include <xtl/xany.hpp>

#include "xtensor.hpp"

template <typename T>
T strtot(char* s)
{
    static std::stringstream ss;
    T t;

    ss.str("");
    ss.clear();

    ss << s;
    ss >> t;

    return t;
}

int main(int argc, char* argv[])
{
    size_t width, height, dim, size;

    if (argc < 3) {
        std::cout << "no width/height given\n";
        return EXIT_FAILURE;
    }

    width = strtot<size_t>(argv[1]);
    height = strtot<size_t>(argv[2]);
    dim = width * height;
    size = dim;
    std::cout << width << '\n' << height << '\n' << dim << '\n' << size << '\n';

    uint8_t* buf = (uint8_t*)amalloc(size);
    read(STDIN_FILENO, buf, size);

    xt::static_shape<size_t, 2> shape = { width, height };
    auto arr = xt::adapt(buf, size, false, shape);
    std::cout << arr << '\n';
}
