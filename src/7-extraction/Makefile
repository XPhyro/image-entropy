.DEFAULT_GOAL := compile

include config.mk

VPATH = src # not POSIX

CXXFLAGS = -O3 -DNDEBUG -std=c++2b \
           -Wall -Wextra -Werror -Wabi=11 \
           -Wno-unused-parameter -Wno-unused-result \
           -Wno-implicit-fallthrough -Wno-sign-compare -Wstringop-overflow=4 \
           -Wfloat-equal -Wdouble-promotion -Wdisabled-optimization

EXEC = extract
SRC = main.cpp
OBJ = $(SRC:.cpp=.o)

compile: $(EXEC)

$(OBJ): %.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $< -o $@

$(EXEC): $(OBJ)
	$(CXX) $^ -o $(EXEC)

clean:
	rm -f -- $(OBJ) $(EXEC)
