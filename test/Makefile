ROOT_DIR = ..
INC = $(ROOT_DIR)/include
BIN = $(ROOT_DIR)/bin
LIBS = -lminim -lellib

CXX = g++
CXXFLAGS = -I$(INC) -L$(BIN) $(LIBS) -DPARALLEL

TESTS = Bitss_test
RUN_TESTS = $(addprefix run_, $(TESTS))

.PHONY: all clean $(RUN_TESTS) gtest

all: $(TESTS) $(RUN_TESTS) clean

clean:
	rm -f $(TESTS) gtest.a gtest_main.a *.o

$(RUN_TESTS): run_%: %
	./$<

# Googletest rules and parameters, see:
# https://github.com/google/googletest/blob/release-1.8.1/googletest/make/Makefile
GTEST_DIR = googletest/googletest
CPPFLAGS += -isystem $(GTEST_DIR)/include
GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h $(GTEST_DIR)/include/gtest/internal/*.h
GTEST_SRCS_ = $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS)

gtest:
	git submodule update --init $(dir $(GTEST_DIR))

gtest-all.o: $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c $(GTEST_DIR)/src/gtest-all.cc

gtest_main.o: $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c $(GTEST_DIR)/src/gtest_main.cc

gtest.a: gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

gtest_main.a: gtest-all.o gtest_main.o
	echo $(GTEST_HEADERS)
	$(AR) $(ARFLAGS) $@ $^

# Tests
$(TESTS): %_test: %_test.o gtest_main.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -lpthread -o $@

%_test.o: %_test.cpp $(GTEST_HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $<
