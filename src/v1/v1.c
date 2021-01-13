#include <stdio.h>
#include <stdint.h>
#include "v0.h"
#include "v1.h"
#include "utils.h"
#include <mpi.h>
#include <math.h>
#include <string.h>

/*
 * Ring-wise communication between processes
 *
 * Otherwise, rank 0 is the master and all the others the slaves that listening only the master
 */
#define RING

#define RANDOM

knnresult distrAllkNN(double *x, uint32_t n, uint32_t d, uint32_t k, int argc, char *argv[]) {

    struct timespec tic;
    struct timespec toc;

    int numtasks, rank, errc;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&numtasks);

    knnresult ret_master;

    if (rank == MASTER) {
        printf(CYN "==>" RESET " Running on " RED "MASTER" RESET " process...\n");
        printf(CYN "==>" RESET " Total number of processes: %d\n", numtasks);
        printf("%d\n", numtasks);
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

#ifdef RANDOM
        MALLOC(double, x, n*d);

        //srand(time(NULL));
        srand(1);

        //printf("Random generated corpus...\n");
        for (uint32_t i = 0; i< n; i++) {
            for (uint32_t j = 0; j < d; j++) {
                x[i*d + j] = (double)(rand()%100);
            }
        }
#else
        x = read_matrix(&n, &d, argc, argv);
#endif
        //printf("\nThe corpus set: \n");
        //print_dataset_yav(x, n, d);        

        ret_master.k = k;
        ret_master.m = n;
        MALLOC(double, ret_master.ndist, n*k);
        MALLOC(uint32_t, ret_master.nidx, n*k);

        printf("n = %d, d = %d, k = %d\n", n, d, k);
        //printf("%d\n", k);
        printf("Corpus array size: %0.3lf MB\n", n*d*8/1e6);
        printf("Total distance matrix size: %0.3lf MB\n", n*n*8/1e6);

    }

#ifndef RANDOM
    MPI_Bcast(&n, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&d, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
#endif

    int corpus_per_proc[numtasks]; 
    int distance_offset[numtasks]; 
    memdistr(n, d, numtasks, corpus_per_proc, distance_offset); // configure the processes' memory chunks of the nxd corpus set

    /* if (rank == MASTER){ */
	    /* printf("The size per process\n"); */
	    /* for(int i = 0; i < numtasks; i++) printf("%d ", corpus_per_proc[i]); */
	    /* printf("\n"); */
	    /* printf("The distance_offset per process\n"); */ 
	    /* for(int i = 0; i < numtasks; i++) printf("%d ", distance_offset[i]); */ 
    /* } */

    double *init_buffer;
    MALLOC(double, init_buffer, corpus_per_proc[rank]);
    MPI_Scatterv(x, corpus_per_proc, distance_offset, MPI_DOUBLE, init_buffer, corpus_per_proc[rank], MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    /* printf("\nRank %d got %d points from corpus\n", rank, corpus_per_proc[rank]/d); */
    /* print_dataset_yav(init_buffer, corpus_per_proc[rank]/d, d); */
    /* printf("\n"); */

    int kNN_per_proc[numtasks];
    int kNN_offset[numtasks];
    memdistr(n, k, numtasks, kNN_per_proc, kNN_offset); // configure the processes' memory chunks of the nxk (distance or indeces)

    /* if (rank == MASTER){ */
    /* printf("\nThe size per process\n"); */
    /* for(int i = 0; i < numtasks; i++) printf("%d ", kNN_per_proc[i]); */
    /* printf("\n"); */
    /* printf("The kNN_offset per process\n"); */
    /* for(int i = 0; i < numtasks; i++) printf("%d ", kNN_offset[i]); */
    /* } */
    /* printf("\n"); */

    
    int m_per_process = corpus_per_proc[rank]/d;

    knnresult ret_per_process;
    ret_per_process.m = m_per_process;
    MALLOC(double, ret_per_process.ndist, m_per_process * k);
    MALLOC(uint32_t, ret_per_process.nidx, m_per_process * k);

    for (int i = 0; i < m_per_process * k; i++) {
        ret_per_process.ndist[i] = INFINITY;
        ret_per_process.nidx[i] = UINT32_MAX;
    }

    double *send_buffer;
    double *curr_buffer;
    double *next_buffer;
    knnresult ret_curr;

#ifndef RING
    MPI_Request req;
    MPI_Status stat;
#else
    MPI_Request reqs[2];
    MPI_Status stats[2];
    int tag = 1;
#endif

    TIC()

    for (int i = 0; i < numtasks; i++) {
        int isSend = 0;

        int offset = distance_offset[rank]/d;
        int local_n = corpus_per_proc[rank]/d;

        // first time get data via scatterv
        if (i == 0) {
            // do not free init buffer
            MALLOC(double, curr_buffer, corpus_per_proc[rank]);
            memcpy(curr_buffer, init_buffer, sizeof(double) * corpus_per_proc[rank]);
        }
        else {
            MALLOC(double, curr_buffer, corpus_per_proc[rank]);
            memcpy(curr_buffer, next_buffer, sizeof(double) * corpus_per_proc[rank]);
            free(next_buffer);
        }

        // one step ahead. Calculate the next buffer to avoid communication cost
        if (i <= numtasks-2) {

#ifdef RING
            int prev = rank-1;
            int next = rank+1;
            if (rank == 0)  prev = numtasks - 1;
            if (rank == (numtasks - 1))  next = 0; 

            MALLOC(double, send_buffer, corpus_per_proc[rank]);
            memcpy(send_buffer, curr_buffer, sizeof(double) * corpus_per_proc[rank]);

            //printf("Rank: %d. The send buff is:\n", rank);
            //print_dataset_yav(send_buffer, local_n, d);

            MPI_Isend(send_buffer, corpus_per_proc[rank], MPI_DOUBLE, prev, tag, MPI_COMM_WORLD, &reqs[0]);
#endif
            // cycle through the size per proc to fill the distance matrix
            rotate_left(corpus_per_proc, numtasks);
            rotate_left(distance_offset, numtasks);

            MALLOC(double, next_buffer, corpus_per_proc[rank]);
#ifndef RING
            MPI_Iscatterv(x, corpus_per_proc, distance_offset, MPI_DOUBLE, next_buffer, corpus_per_proc[rank], MPI_DOUBLE, MASTER, MPI_COMM_WORLD, &req);
#else
            // TODO Exclude the master process to receive because it already has the data
            MPI_Irecv(next_buffer, corpus_per_proc[rank], MPI_DOUBLE, next, tag, MPI_COMM_WORLD, &reqs[1]);
#endif
            isSend = 1;
        }


        ret_curr = kNN(curr_buffer, init_buffer, local_n, m_per_process, d, k);
        free(curr_buffer);
        //printf("\n");

        /* printf("\nIndeces of kNN ret curr before\n"); */
        /* print_indeces(ret_curr.nidx, ret_curr.m, k); */

        adjust_indeces(ret_curr.nidx, m_per_process, MIN(local_n, k), offset);

        /* printf("\nDistance of kNN ret curr\n"); */
        /* print_dataset_yav(ret_curr.ndist, ret_curr.m, k); */
        /* printf("\nIndeces of kNN ret curr after\n"); */
        /* print_indeces(ret_curr.nidx, ret_curr.m, k); */

        /* if (i != 0) { */
        /*     printf("\nDistance of kNN ret old\n"); */
        /*     print_dataset_yav(ret_per_process.ndist, ret_per_process.m, k); */
        /*     printf("\nIndeces of kNN ret old\n"); */
        /*     print_indeces(ret_per_process.nidx, ret_per_process.m, k); */
        /* } */

        // compare the old data with the newest data
        if (i != 0) {
            for (int j = 0; j < m_per_process; j++) {

                int len_curr = MIN(local_n, k);
                int len_total = k + len_curr;

                double *merged_distance;
                MALLOC(double, merged_distance, len_total); 

                memcpy(merged_distance, ret_per_process.ndist + j*k, sizeof(double) * k);
                /* printf("The ret old\n"); */
                /* for(int l = 0; l < k; l++) printf("%lf ", ret_per_process.ndist[l + j*k]); */
                /* printf("\n"); */
                /* printf("The merged distance is\n"); */
                /* for(int l = 0; l < k; l++) printf("%lf ", merged_distance[l]); */
                /* printf("\n"); */
                memcpy(merged_distance + k, ret_curr.ndist + j*len_curr, sizeof(double) * len_curr);
                /* printf("The merged distance is\n"); */
                /* for(int l = 0; l < len_total; l++) printf("%lf ", merged_distance[l]); */
                /* printf("\n"); */

                uint32_t *merged_indeces;
                MALLOC(uint32_t, merged_indeces, len_total); 

                memcpy(merged_indeces, ret_per_process.nidx + j*k, sizeof(uint32_t) * k);
                /* printf("The merged indeces is\n"); */
                /* for(int l = 0; l < k; l++) printf("%d ", merged_indeces[l]); */
                /* printf("\n"); */
                memcpy(merged_indeces + k, ret_curr.nidx + j*len_curr, sizeof(uint32_t) * len_curr);
                /* printf("The merged indeces is\n"); */
                /* for(int l = 0; l < len_total; l++) printf("%d ", merged_indeces[l]); */
                /* printf("\n"); */

                // update
                qselect_and_indeces(merged_distance, merged_indeces, len_total, k-1);
                memcpy(ret_per_process.ndist + j*k, merged_distance, sizeof(double) * k);
                memcpy(ret_per_process.nidx + j*k, merged_indeces, sizeof(uint32_t) * k);

                /* printf("The per process distance is\n"); */
                /* for(int l = 0; l<k; l++) printf("%lf ", ret_per_process.ndist[l + j*k]); */
                /* printf("\n"); */

                /* printf("The per process indeces is\n"); */
                /* for(int l = 0; l<k; l++) printf("%d ", ret_per_process.nidx[l + j*k]); */
                /* printf("\n"); */

                free(merged_distance);
                free(merged_indeces);
            }

            free(ret_curr.ndist);
            free(ret_curr.nidx);
        }
        else {
            for (int j = 0; j < m_per_process; j++) {
                memcpy(ret_per_process.ndist + j*k, ret_curr.ndist + j*MIN(local_n,k), sizeof(double) * MIN(local_n, k));
                memcpy(ret_per_process.nidx + j*k, ret_curr.nidx + j*MIN(local_n, k), sizeof(uint32_t) * MIN(local_n, k));
            }
            free(ret_curr.ndist);
            free(ret_curr.nidx);
        }

        if(isSend) {
            //TIC()
#ifndef RING
            MPI_Wait(&req, &stat);
#else
            MPI_Waitall(2, reqs, stats);
            free(send_buffer);
            //printf("Rank: %d. The next buff is:\n", rank);
            //print_dataset_yav(next_buffer, corpus_per_proc[rank]/d, d);
#endif
            //TOC(RED "Cost " RESET "for syncing: %lf\n")
        }

        /* if(i <= numtasks-2) { */
        /*     printf("Rank: %d. The next buff is:\n", rank); */
        /*     print_dataset_yav(next_buffer, corpus_per_proc[rank]/d, d); */
        /* } */

    }

    free(init_buffer);
    if(rank == MASTER) free(x);

    //printf("\n");

    /* printf("\nRank %d. Distance of kNN\n", rank); */
    /* print_dataset_yav(ret_per_process.ndist, ret_per_process.m, k); */
    /* printf("\nRank %d. Indeces of kNN\n", rank); */
    /* print_indeces(ret_per_process.nidx, ret_per_process.m, k); */

    clock_gettime(CLOCK_MONOTONIC, &toc);
    //printf(CYN "Rank: %d. " RESET "Time elapsed calculating per process kNN (seconds): %lf\n", rank, diff_time(tic, toc));

    //if(rank == MASTER) {TIC();}

    // Gather to root
    MPI_Gatherv(ret_per_process.ndist, kNN_per_proc[rank], MPI_DOUBLE, ret_master.ndist, kNN_per_proc, kNN_offset, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Gatherv(ret_per_process.nidx, kNN_per_proc[rank], MPI_INT, ret_master.nidx, kNN_per_proc, kNN_offset, MPI_INT, MASTER, MPI_COMM_WORLD);

    //if(rank == MASTER) {TOC(RED "\nCOST " RESET "Gatherv: %lf\n")}

    free(ret_per_process.ndist);
    free(ret_per_process.nidx);

    return ret_master;
}
