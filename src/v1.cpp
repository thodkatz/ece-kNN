#include <stdio.h>
#include <stdint.h>
#include "include/main.h"
#include <mpi.h>


knnresult distrAllkNN(double * x, uint32_t n, uint32_t d, uint32_t k) {

    knnresult ret;
    ret.k = k;
    ret.m = n;
    MALLOC(double, ret.ndist, n*k);
    MALLOC(uint32_t, ret.nidx, n*k);

    printf("Helloooo\n");


    return ret;
}

