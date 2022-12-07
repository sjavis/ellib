TARGET = libellib.a
SRC_DIR = src
INC_DIR = include
BUILD_DIR = bin
LIB_DIR = lib
LIBS = minim

CXX      = mpicxx#            C++ compiler
CXXFLAGS = -Wall -DPARALLEL#  Flags for the C++ compiler

TARGET := $(BUILD_DIR)/$(TARGET)
VPATH = $(SRC_DIR)
SRC = $(foreach sdir, $(SRC_DIR), $(wildcard $(sdir)/*.cpp))
OBJ = $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(notdir $(SRC)))
INC = $(addprefix -I, $(INC_DIR))
LIB = $(patsubst %,$(BUILD_DIR)/lib%.a, $(LIBS))
LDLIBS = $(addprefix -l, $(LIBS))
LDFLAGS = $(addprefix -L, $(BUILD_DIR))

.PHONY: all debug deps clean check $(LIB)

all: $(TARGET)

debug: CXXFLAGS += -g
debug: SUBTARGET = debug
debug: $(TARGET)

deps: $(LIB)

clean:
	rm $(TARGET) $(OBJ) $(LIB)

$(TARGET): $(OBJ)
	ar -rcs $(TARGET) $(OBJ)

$(OBJ): $(BUILD_DIR)/%.o: %.cpp $(LIB)
	$(CXX) $(CXXFLAGS) $(INC) -c $< $(LDFLAGS) $(LDLIBS) -o $@

$(LIB): $(BUILD_DIR)/lib%.a:
	@git submodule update --init $(LIB_DIR)/$*
	$(MAKE) -C $(LIB_DIR)/$* $(SUBTARGET)
	@ln -sfn ../$(LIB_DIR)/$*/$(INC_DIR) $(INC_DIR)/$*
	@cp $(LIB_DIR)/$*/$@ $@

check:
	@echo Testing...
	@$(MAKE) --no-print-directory -C test gtest
	@$(MAKE) --no-print-directory -C test
