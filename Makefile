CC=g++
CCMPI=mpic++
CFLAGS= -g -pedantic -Wall
LDFLAGS= -lopenblas

BIN = bin
DIRSRC = src
DIR_v0 = $(DIRSRC)/v0
DIR_v1 = $(DIRSRC)/v1
DIR_v2 = $(DIRSRC)/v2
BASE = $(DIRSRC)/*.c $(DIRSRC)/*.cpp 
SRC_v0 = $(BASE) $(DIR_v0)/*.cpp 
SRC_v1 = $(BASE) $(DIR_v1)/*.cpp $(DIR_v0)/v0.cpp
SRC_v2 = $(BASE) $(DIR_v2)/*.cpp
INC = -I include

all: v0 v1 v2

v0: $(SRC_v0)
	$(CC) $(CFLAGS) $^ $(INC) -o $(BIN)/$@ $(LDFLAGS)

v1: $(SRC_v1)
	$(CCMPI) $(CFLAGS) $^ $(INC) -o $(BIN)/$@ $(LDFLAGS)

v2: $(SRC_v2)
	$(CCMPI) $(CFLAGS) $^ $(INC) -o $(BIN)/$@ $(LDFLAGS)

.PHONY: clean v0 v1 v2

clean:
	rm -f bin/*

