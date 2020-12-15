CC=g++
CFLAGS= -O3 -Wall
LDFLAGS= -lopenblas

SRC = main.cpp diff_time.cpp mm2coo.c mmio.c v0.cpp

all: main
	
main: $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $@ $(LDFLAGS)

.PHONY: clean

clean:
	rm -f main
