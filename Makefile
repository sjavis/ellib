SRC_DIR = src
VPATH = $(SRC_DIR)
SRC = $(foreach sdir, $(SRC_DIR), $(wildcard $(sdir)/*.cpp))
OBJ = $(patsubst %.cpp,bin/%.o,$(notdir $(SRC)))
INC = $(addprefix -I, $(SRC_DIR) include)

LIBS = minim
LDLIBS = $(addprefix -l, $(LIBS))
LIBPATHS = $(patsubst %,bin/lib%.a, $(LIBS))
LDFLAGS = -Lbin

CXX      = mpic++       # C++ compiler
CXXFLAGS = $(INC) $(LDFLAGS) $(LDLIBS) -DPARALLEL # Flags for the C++ compiler

.PHONY: all deps clean check

all: $(OBJ)
	ar -rcs bin/libellib.a $(OBJ)

deps: $(LIBPATHS)

clean:
	rm bin/*.o bin/libellib.a $(LIBPATHS)

$(OBJ): bin/%.o: %.cpp $(LIBPATHS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(LIBPATHS): bin/lib%.a:
	git submodule update --init lib/$*
	$(MAKE) -C lib/$*
	ln -sfn ../lib/$*/include include/$*
	cp lib/$*/$@ $@

check:
	$(MAKE) -C test gtest
	$(MAKE) -C test
