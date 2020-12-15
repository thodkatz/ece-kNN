CC=gcc
CFLAGS=-std=c11 -g -Wall -lm

SRC = main.c diff_time.c coo2csc.c mm2coo.c mmio.c v0.c

all: main
	
main: $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $@ -lopenblas

.PHONY: clean

clean:
	rm -f main
