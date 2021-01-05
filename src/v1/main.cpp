#include <cblas.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <stdlib.h>
#include "main.h"
#include "v1.h"
#include <math.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    /**********************************************************/
    /*                    Version 1                           */
    /**********************************************************/

    int  numtasks, rank; 
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    uint32_t n = (uint32_t)3e4;
    uint32_t d = 4;
    uint32_t k = 3;

    double *x;
    knnresult ret;

    struct timespec tic;
    struct timespec toc;

    FILE *log;

    TIC()

    // the return is meaningful only for the MASTER
    ret = distrAllkNN(x, n, d, k);

    if (rank == MASTER) {

        TOC(RED "\nTOTAL: " RESET "Time elapsed calculating kNN (seconds): %lf\n")

        /* printf("\nDistance of kNN\n"); */
        /* print_dataset_yav(ret.ndist, ret.m, k); */
        /* printf("\nIndeces of kNN\n"); */
        /* print_indeces(ret.nidx, ret.m, k); */

        log = fopen("v1_log.txt", "w");

        // for comparing v1 and v2 sort the data
        for(int j = 0; j < k; j++) {
            for(int i = 0; i < ret.m; i++) {
                qselect(ret.ndist + i*k, ret.nidx + i*k, k, j);
            }
        }

        print_output_file(log, ret.ndist, ret.nidx, ret.m, k);
        fclose(log);

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

void print_output_file(FILE *f, double *dist, uint32_t *indeces, uint32_t row, uint32_t col) {
    for (uint32_t i = 0; i < row; i++)
    {
        if (i == 0)
            fprintf(f, "[[");
        else
            fprintf(f, " [");
        for (uint32_t j = 0; j < col; j++)
            if (j != col -1) fprintf(f, "%lf,", dist[i*col + j]);
            else fprintf(f, "%lf", dist[i*col + j]);
        if (i == row -1)
            fprintf(f, "]]\n");
        else
            fprintf(f, "],\n");
    }

    for (uint32_t i = 0; i < row; i++)
    {
        if (i == 0)
            fprintf(f, "[[");
        else
            fprintf(f, " [");
        for (uint32_t j = 0; j < col; j++) {
            if (j != col -1) fprintf(f, "%u,", indeces[i*col + j]);
            else fprintf(f, "%u", indeces[i*col + j]);
        }
        if (i == row -1)
            fprintf(f, "]]\n");
        else
            fprintf(f, "],\n");
    }
}
