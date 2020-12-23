#include <time.h>
#include <stdint.h>
#include <stdio.h>

double diff_time (struct timespec start, struct timespec end) {
    uint32_t diff_sec = (end.tv_sec - start.tv_sec);
    int32_t diff_nsec = (end.tv_nsec - start.tv_nsec);
    if ((end.tv_nsec - start.tv_nsec) < 0) {
        diff_sec -= 1;
        diff_nsec = 1e9 + end.tv_nsec - start.tv_nsec;
    }

    return (1e9*diff_sec + diff_nsec)/1e9;
}
