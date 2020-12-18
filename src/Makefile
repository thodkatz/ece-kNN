CC=g++
CCMPI=mpic++
CFLAGS= -O3 -pedantic -Wall
LDFLAGS= -lopenblas

DIR = bin/
SRC_v0 = main_v0.cpp diff_time.cpp mm2coo.c mmio.c v0.cpp
SRC_v1 = main_v1.cpp diff_time.cpp mm2coo.c mmio.c v0.cpp v1.cpp

all: v0 v1
	
v0: $(SRC_v0)
	$(CC) $(CFLAGS) $(SRC_v0) -o $(DIR)$@ $(LDFLAGS)

v1: $(SRC_v1)
	$(CCMPI) $(CFLAGS) $(SRC_v1) -o $(DIR)$@ $(LDFLAGS)

.PHONY: clean

clean:
	rm -f v0 v1
