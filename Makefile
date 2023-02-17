TARGET = libellib.a
SRC_DIR = src
INC_DIR = include
BUILD_DIR = bin
LIB_DIR = lib
LIBS = minim

CXX      = mpicxx#            C++ compiler
CXXFLAGS = -Wall -DPARALLEL -std=c++14#  Flags for the C++ compiler

TARGET := $(BUILD_DIR)/$(TARGET)
VPATH = $(SRC_DIR)
SRC = $(foreach sdir, $(SRC_DIR), $(wildcard $(sdir)/*.cpp))
OBJ = $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(notdir $(SRC)))
INC = $(addprefix -I, $(INC_DIR))
LIB = $(patsubst %,$(BUILD_DIR)/lib%.a, $(LIBS))
LDLIBS = $(addprefix -l, $(LIBS))
LDFLAGS = $(addprefix -L, $(BUILD_DIR))

.PHONY: all debug deps clean check $(LIBS)

all: deps $(TARGET)

debug: CXXFLAGS += -g
debug: SUBTARGET = debug
debug: deps $(TARGET)

deps: $(LIBS)

clean:
	rm $(TARGET) $(OBJ) $(LIB)

$(TARGET): $(OBJ) $(LIB)
	@echo "Making library: $@"
	@ar -rcs $@ $(OBJ)
	@echo "CREATE $@" >> tmp.mri
	@echo "ADDLIB $@" >> tmp.mri
	@for LIBFILE in $(LIB); do echo "ADDLIB $$LIBFILE" >> tmp.mri; done
	@echo "SAVE" >> tmp.mri
	@echo "END" >> tmp.mri
	@ar -M < tmp.mri
	@rm tmp.mri

$(OBJ): $(BUILD_DIR)/%.o: %.cpp $(LIB)
	$(CXX) $(CXXFLAGS) $(INC) -c $< $(LDFLAGS) $(LDLIBS) -o $@

$(LIBS): %:
	@git submodule update --init $(LIB_DIR)/$*
	$(MAKE) --no-print-directory -C $(LIB_DIR)/$* $(SUBTARGET)
	@ln -sfn ../$(LIB_DIR)/$*/$(INC_DIR) $(INC_DIR)/$*
	@if [ ! -f $(BUILD_DIR)/lib$*.a ] || [ $(LIB_DIR)/$*/$(BUILD_DIR)/lib$*.a -nt $(BUILD_DIR)/lib$*.a ]; then\
	  cp $(LIB_DIR)/$*/$(BUILD_DIR)/lib$*.a $(BUILD_DIR)/lib$*.a;\
	fi

$(LIB): $(BUILD_DIR)/lib%.a:
	$(MAKE) --no-print-directory $*

check:
	@echo Testing...
	@$(MAKE) --no-print-directory -C test gtest
	@$(MAKE) --no-print-directory -C test
