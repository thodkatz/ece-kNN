#include <cblas.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <stdlib.h>
#include "v0.h"
#include "utils.h"
#include <math.h>
#include <string.h>

#define BLOCKS 1

/*
 * ifndef --> query subset of corpus 
 * ifdef --> distrAll
 */
#define ALL 

/*
 * ifndef --> take input a dataset
 * ifdef  --> take input a random matrix
 */
#define RANDOM

int main(int argc, char *argv[])
{

    /**********************************************************/
    /*                    Version 0                           */
    /**********************************************************/

    //printf("\n<----------Version 0---------->\n");

    uint32_t n = 1e2;
    uint32_t d = 4;
    uint32_t m = 8;
    uint32_t k = 10;

    if(argc < 2) {
        printf("Bad arguments\n");
        exit(-1);
    }

    k = atoi(argv[2]);

    if (m > n) {
        printf("Number of query elements exceeded the number of elements in the corpus set\n");
        return -1;
    }
    if (k > n) {
        printf("Number of nearest elements exceeded the number of elements in the corpus set\n");
        k = n;
    }

    struct timespec tic;
    struct timespec toc;

    //srand(time(NULL));
    srand(1);


    double *x, *y;
#ifdef RANDOM
    printf("Random generated corpus...\n");

    MALLOC(double, x, n*d);
    MALLOC(double, y, m*d);

    for (uint32_t i = 0; i< n; i++) {
        for (uint32_t j = 0; j < d; j++) {
            x[i*d + j] = (double)(rand()%100);
        }
    }
#else
    //TIC()
    printf("Reading %s\n", argv[1]);
    x = read_matrix(&n, &d, argc, argv);
    //TOC("Time elasped reading corpus: %lf\n")
#endif

#ifdef ALL
    m = n; // symmetric
#endif

    //printf("%u\n", k);
    printf("n = %u, d = %u, m = %u, k = %u\n", n, d, m, k);
    printf("Corpus array size: %0.3lf MB\nQuery array size: %0.3lf MB\n", n*d*8/1e6, m*d*8/1e6);

    knnresult ret;
    ret.k = k;
    ret.m = m;
    MALLOC(double, ret.ndist, m*k);
    MALLOC(uint32_t, ret.nidx, m*k);

    // blocking the query
    
    uint32_t blocks = BLOCKS;
    if (blocks == 0) blocks = 1;
    printf("Blocking the query into %d blocks\n", blocks);
    uint32_t block_size = m/blocks; 
    printf("Each block has %u points\n", block_size);

    FILE *input;
    input = fopen("logs/input.txt", "w");
    print_input_file(input, x, n, d);
    fclose(input);
    /* print_dataset_yav(x, n, d); */ 

    TIC()

    // for each block (subset of query set) calculate the kNN and then merge the results

    for (uint32_t curr = 0; curr < blocks; curr++) {
        uint32_t start = curr*block_size;
        uint32_t end = start + block_size;
        if (curr == blocks-1) end = m; // FIX not good load balance. The remaining should be given like a deck of cards
        //printf("\nRange for %uth iteration: [%u, %u]\n", curr, start, end);
        //printf("Random generated query...\n");
#ifndef ALL
        for (uint32_t i = start; i< end; i++) {
            uint32_t random = rand()%n; // there are duplicate indexes
            for (uint32_t j = 0; j < d; j ++) {
                y[i*d + j] = x[random*d + j];
            }
        }
#endif
    
    /* print_dataset(x, n, d); */
    /* printf("\n"); */
    /* print_dataset_yav(y + start*d, end-start, d); */
    
    knnresult ret_blocked;
#ifndef ALL
    ret_blocked = kNN(x, y + start*d, n, end-start, d, k);
#else
    ret_blocked = kNN(x, x + start*d, n, end-start, d, k);
#endif
    memcpy(ret.ndist + start*k, ret_blocked.ndist, sizeof(double) * k * (end - start));
    memcpy(ret.nidx + start*k, ret_blocked.nidx, sizeof(uint32_t) * k * (end - start));

    free(ret_blocked.ndist);
    free(ret_blocked.nidx);
    }

    TOC("\nTime elapsed calculating kNN total (seconds): %lf\n")
    //TOC("%lf\n")

    /* printf("\nDistance of kNN\n"); */
    /* print_dataset_yav(ret.ndist, m, k); */
    /* printf("\nIndeces of kNN\n"); */
    /* print_indeces(ret.nidx, m, k); */

    FILE *log;
    log = fopen("logs/v0_log.txt", "w");

    // for comparing v1 and v2 sort the data
    for(int j = 0; j < k; j++) {
        for(int i = 0; i < ret.m; i++) {
            qselect_and_indeces(ret.ndist + i*k, ret.nidx + i*k, k, j);
       }
    }
    print_output_file(log, ret.ndist, ret.nidx, m, k);
    fclose(log);

    free(x);
#ifndef ALL
    free(y);
#endif
    free(ret.nidx);
    free(ret.ndist);

    return 0;
}
