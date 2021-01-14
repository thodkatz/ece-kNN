#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "v0.h"
#include "v2.h"
#include "utils.h"
#include "Vptree.h"

knnresult kNN_vptree(Vptree &vpt, double *y, int n, uint32_t m, uint32_t d, uint32_t k) {

    knnresult ret;
    MALLOC(double, ret.ndist, m*MIN(k, n));
    MALLOC(uint32_t, ret.nidx, m*MIN(k, n));
    ret.k = MIN(n,k); 
    ret.m = m;

    struct timespec tic;
    struct timespec toc;
    
    TIC()

    // search vp tree
    for(uint32_t i = 0; i < m; i++) {
        double *target;
        MALLOC(double, target, d);
        memcpy(target, y + i*d, sizeof(double) * d);
        vpt.searchKNN(ret.ndist + MIN(k,n)*i, ret.nidx + MIN(k,n)*i, target, MIN(n,k));
        free(target);

    }

    //TOC("Time elapsed calculating kNN vp-tree (seconds): %lf\n")
    return ret;
}
