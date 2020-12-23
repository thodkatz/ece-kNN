CC=g++
CCMPI=mpic++
CFLAGS= -g -pedantic -Wall
LDFLAGS= -lopenblas

BIN = bin/
BASE = diff_time.cpp mm2coo.c mmio.c v0.cpp
v0 = $(BASE) main_v0.cpp
v1 = $(BASE) main_v1.cpp v1.cpp

all: v0 v1
	
v0: $(v0)
	$(CC) $(CFLAGS) $(v0) -o $(BIN)$@ $(LDFLAGS)

v1: $(v1)
	$(CCMPI) $(CFLAGS) $(v1) -o $(BIN)$@ $(LDFLAGS)

.PHONY: clean

clean:
	rm -f bin/*

