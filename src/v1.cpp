#include <stdio.h>
#include <stdint.h>
#include "include/main.h"
#include <mpi.h>

#define MASTER 0

// color because why not
#define CYN   "\x1B[36m"
#define RED   "\x1B[31m"
#define RESET "\x1B[0m"

extern void print_dataset(double *array, uint32_t row, uint32_t col);
extern void print_dataset_yav(double *array, uint32_t row, uint32_t col);
extern void print_indeces(uint32_t *array, uint32_t row, uint32_t col);

knnresult distrAllkNN(double *x, uint32_t n, uint32_t d, uint32_t k) {

    int numtasks, rank, errc;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&numtasks);

    knnresult ret_master;

    if (rank == MASTER) {
        printf(CYN "==>" RESET " Running on " RED "MASTER" RESET " process...\n");
        printf(CYN "==>" RESET " Total number of processes: %d\n", numtasks);
        printf(CYN "==>" RESET " VERSION 1\n");

        if (k > n) {
            printf(RED "Error: " RESET "Number of nearest elements exceeded the number of elements in the corpus set\n");
            errc = -1;
            MPI_Abort(MPI_COMM_WORLD, errc);
        }
        if (numtasks > n) {
            printf(RED "Error: " RESET "Number of processes exceeded the number of elements in the corpus set\n");
            errc = -1;
            MPI_Abort(MPI_COMM_WORLD, errc);
        }

        ret_master.k = k;
        ret_master.m = n;
        MALLOC(double, ret_master.ndist, n*k);
        MALLOC(uint32_t, ret_master.nidx, n*k);

        //srand(time(NULL));
        srand(1);
        printf("n = %u, d = %u, k = %u\n", n, d, k);
        printf("Corpus array size: %0.3lf MB\n", n*d*8/1e6);
        printf("Random generated corpus...\n");

        MALLOC(double, x, n*d);

        for (uint32_t i = 0; i< n; i++) {
            for (uint32_t j = 0; j < d; j++) {
                x[i*d + j] = (double)(rand()%100);
            }
        }

        /* printf("\nThe corpus set: \n"); */
        /* print_dataset_yav(x, n, d); */        
    }

    int size_per_proc[numtasks];
    int memory_offset[numtasks]; 
    int remain = n%numtasks;
    for (int i = 0; i < numtasks; i++) {
        memory_offset[i] = 0;

        size_per_proc[i] = n/numtasks * d;

        // the remaining n share them like a deck of cards (better load balance)
        if (remain) {
            size_per_proc[i] += d;
            remain--;
        }

        if(i!= 0) memory_offset[i] = memory_offset[i-1] + size_per_proc[i-1];
    }


    if (rank == MASTER){
    printf("The size per process\n");
    for(int i = 0; i < numtasks; i++) printf("%d ", size_per_proc[i]);
    printf("\n");
    printf("The memory_offset per process\n");
    for(int i = 0; i < numtasks; i++) printf("%d ", memory_offset[i]);
    }

    double *working_buffer;
    //double *remaining_buffer;
    MALLOC(double, working_buffer, size_per_proc[rank]);
    //MALLOC(double, remaining_buffer, size_per_proc[rank]);

    MPI_Scatterv(x, size_per_proc, memory_offset, MPI_DOUBLE, working_buffer, size_per_proc[rank], MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    printf("\nRank %d got %d points from corpus\n", rank, size_per_proc[rank]/d);
    print_dataset_yav(working_buffer, size_per_proc[rank]/d, d);
    printf("\n");

    knnresult ret_working;
    ret_working.m = size_per_proc[rank]/d;
    ret_working.k = k;
    MALLOC(double, ret_working.ndist, ret_working.m * k);
    MALLOC(uint32_t, ret_working.nidx, ret_working.m * k);

    ret_working = kNN(working_buffer, working_buffer, ret_working.m, ret_working.m, d, k);
    printf("\n");

    /* printf("\nDistance of kNN\n"); */
    /* print_dataset_yav(ret_working.ndist, ret_working.m, k); */
    /* printf("\nIndeces of kNN\n"); */
    /* print_indeces(ret_working.nidx, ret_working.m, k); */


    // Gather to root
    // gatherv

    free(working_buffer);


    return ret_master;
}

