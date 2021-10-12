SRC_DIR = src
SRC = $(foreach sdir, $(SRC_DIR), $(wildcard $(sdir)/*.cpp))
OBJ = $(patsubst %.cpp,bin/%.o,$(notdir $(SRC)))
INC = $(addprefix -I, $(SRC_DIR) include)

LIBS = minim
LDLIBS = $(addprefix -l, $(LIBS))
LIBPATHS = $(patsubst %,bin/lib%.a, $(LIBS))
LDFLAGS = -Lbin

CXX      = mpic++       # C++ compiler
CXXFLAGS = $(INC) $(LDFLAGS) $(LDLIBS) -DPARALLEL # Flags for the C++ compiler

all: $(OBJ)
	ar -rcs bin/libellib.a $(OBJ)

bin/%.o: %.cpp $(LIBPATHS)
	$(CXX) $(CXXFLAGS) -c $^ -o $@

deps: $(LIBPATHS)

bin/lib%.a:
	$(MAKE) -C lib/$*
	ln -sfn ../lib/$*/include include/$*
	cp lib/$*/$@ $@

clean:
	rm bin/*.o bin/libellib.a $(LIBPATHS)
