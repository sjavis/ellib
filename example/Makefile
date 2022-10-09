ROOT_DIR = ..
INC = $(ROOT_DIR)/include
BIN = $(ROOT_DIR)/bin
LIBS = -lellib -lminim

CXX = mpic++
CXXFLAGS = -I$(INC) -L$(BIN) $(LIBS) -Wall -DPARALLEL -g

all:
	$(CXX) example.cpp -o run.exe $(CXXFLAGS)

clean:
	rm run.exe
