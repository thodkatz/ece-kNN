#include <cblas.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <stdlib.h>
#include "include/main.h"
#include <math.h>
#include <string.h>
#include <mpi.h>

#define MASTER 0

double diff_time (struct timespec start, struct timespec end);
void print_dataset(double *array, uint32_t row, uint32_t col);
void print_dataset_yav(double *array, uint32_t row, uint32_t col);
void print_indeces(uint32_t *array, uint32_t row, uint32_t col);

int main(int argc, char *argv[])
{



    /**********************************************************/
    /*                    Version 1                           */
    /**********************************************************/


    int  numtasks, rank; 
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    uint32_t n = (uint32_t)1e1;
    uint32_t d = 4;
    uint32_t m = 8;
    uint32_t k = 3;

    double *x;
    knnresult ret;

    struct timespec tic;
    struct timespec toc;

    TIC();

    // the return is meaningful only for the MASTER
    ret = distrAllkNN(x, n, d, k);

    TOC("Time elapsed calculating kNN per process (seconds): %lf\n");

    if (rank == MASTER) {
        /* printf("\nDistance of kNN\n"); */
        /* print_dataset_yav(ret.ndist, m, k); */
        /* printf("\nIndeces of kNN\n"); */
        /* print_indeces(ret.nidx, m, k); */    

        TOC("\nTime elapsed calculating kNN total (seconds): %lf\n");

        printf("\nDistance of kNN\n");
        //print_dataset_yav(ret.ndist, ret.m, k);
        /* printf("\nIndeces of kNN\n"); */
        /* print_indeces(ret.nidx, ret.m, k); */

        //free(x);
        free(ret.nidx);
        free(ret.ndist);

    }
    
    MPI_Finalize();

    return 0;
}

// print 2d array in matlab format (row wise)
void print_dataset(double *array, uint32_t row, uint32_t col) {
    printf("[ ");
    for (uint32_t i = 0; i < row; i++) {
        for (uint32_t j = 0; j < col; j++) {
            printf("%lf ", array[i*col + j]);
        }
        if (i != row-1) printf("; ");
        else printf("]\n");
    }
}

// print 2d array in python format (row wise)
void print_dataset_yav(double *array, uint32_t row, uint32_t col) {
    for (uint32_t i = 0; i < row; i++)
    {
        if (i == 0)
            printf("[[");
        else
            printf(" [");
        for (uint32_t j = 0; j < col; j++)
            if (j != col -1) printf("%lf,", array[i*col + j]);
            else printf("%lf", array[i*col + j]);
        if (i == row -1)
            printf("]]\n");
        else
            printf("],\n");
    }

}

void print_indeces(uint32_t *array, uint32_t row, uint32_t col) {
    for (uint32_t i = 0; i < row; i++)
    {
        if (i == 0)
            printf("[[");
        else
            printf(" [");
        for (uint32_t j = 0; j < col; j++) {
            if (j != col -1) printf("%u,", array[i*col + j]);
            else printf("%u", array[i*col + j]);
        }
        if (i == row -1)
            printf("]]\n");
        else
            printf("],\n");
    }
}
