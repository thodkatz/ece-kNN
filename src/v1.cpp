#include <stdio.h>
#include <stdint.h>
#include "include/main.h"
#include <mpi.h>
#include <math.h>

#define MASTER 0


#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

// color because why not
#define CYN   "\x1B[36m"
#define RED   "\x1B[31m"
#define RESET "\x1B[0m"

extern void print_dataset(double *array, uint32_t row, uint32_t col);
extern void print_dataset_yav(double *array, uint32_t row, uint32_t col);
extern void print_indeces(uint32_t *array, uint32_t row, uint32_t col);
extern double qselect(double *v, uint32_t *idx, int64_t len, int64_t k);

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

    double *init_buffer;
    MALLOC(double, init_buffer, size_per_proc[rank]);

    MPI_Scatterv(x, size_per_proc, memory_offset, MPI_DOUBLE, init_buffer, size_per_proc[rank], MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    /* printf("\nRank %d got %d points from corpus\n", rank, size_per_proc[rank]/d); */
    /* print_dataset_yav(init_buffer, size_per_proc[rank]/d, d); */
    /* printf("\n"); */

    double *curr_buffer;
    double *new_buffer;
    int next = rank;
    int prev = 0;
    int local_n_curr = 0;
    int local_n_old = 0;

    int m_per_process = size_per_proc[rank]/d;

    // Gather prerequisites, but we need to pass struct, not only dist and idx (2 Gathers required)
    int size_per_proc_final[numtasks];
    int memory_offset_final[numtasks];
    int remain_final = n%numtasks;

    for (int i = 0; i < numtasks; i++) {
        memory_offset_final[i] = 0;

        size_per_proc_final[i] = n/numtasks * k;

        // the remaining n share them like a deck of cards (better load balance)
        if (remain_final) {
            size_per_proc_final[i] += k;
            remain_final--;
        }

        if(i!= 0) memory_offset_final[i] = memory_offset_final[i-1] + size_per_proc_final[i-1];

        /* // dist and idx */
        /* memory_offset_final[i] *= 2; */
        /* size_per_proc_final[i] *= 2; */
    }

    if (rank == MASTER){
    printf("\nThe size per process\n");
    for(int i = 0; i < numtasks; i++) printf("%d ", size_per_proc_final[i]);
    printf("\n");
    printf("The memory_offset_final per process\n");
    for(int i = 0; i < numtasks; i++) printf("%d ", memory_offset_final[i]);
    }
    printf("\n");

    knnresult ret_old;
    knnresult ret_curr;

    knnresult ret_per_process;
    ret_per_process.m = m_per_process;
    MALLOC(double, ret_per_process.ndist, m_per_process * k);
    MALLOC(uint32_t, ret_per_process.nidx, m_per_process * k);

    for (int i = 0; i < m_per_process * k; i++) {
        ret_per_process.ndist[i] = INFINITY;
        ret_per_process.nidx[i] = INFINITY;
    }

    MPI_Request req;
    MPI_Status stat;

    for (int i = 0; i < numtasks; i++) {


        // first time get data via scatterv
        if (i == 0) curr_buffer = init_buffer;
        else curr_buffer = new_buffer;

        // one step ahead. Calculate the next buffer to avoid communication cost
        prev = next;
        int grid = memory_offset[rank]/d;

        if (i <= numtasks-2) {
            next++;
            if (next == numtasks)  next = 0;

            /* // receive */
            MALLOC(double, new_buffer, size_per_proc[next]);

            /* //MPI_Irecv(&new_buffer, size_per_proc[next], MPI_DOUBLE, MASTER, tag, MPI_COMM_WORLD, &req); */
        
            /* // master send data to everyone */
            /* if (rank == MASTER) { */
            /*     for (int j = 0; j < numtasks; j++) { */
            /*         double *local_x; */
            /*         MALLOC(double, local_x, size_per_proc[]); */
            /*         memcpy(local_x, x + memory_offset[next], sizeof(double) * size_per_proc[rank]); */
            /*         // non blocking send */
            /*         MPI_Isend(&local_x, size_per_proc[next], MPI_DOUBLE, rank, tag, MPI_COMM_WORLD, req); */
            /*     } */
            /* } */

            // non blocking scatterv
            // cycle through the size per proc to fill the distance matrix

            // shift size_per_proc and memory_offset
            // rotate left
            double temp1 = size_per_proc[0];
            double temp2 = memory_offset[0];
            for (int k = 0; k < numtasks - 1; k++){        
                size_per_proc[k] = size_per_proc[k+1];
                memory_offset[k] = memory_offset[k+1];
            }
            size_per_proc[numtasks-1] = temp1;
            memory_offset[numtasks-1] = temp2;

            /* if (rank == MASTER){ */
            /*     printf("The size per process\n"); */
            /*     for(int i = 0; i < numtasks; i++) printf("%d ", size_per_proc[i]); */
            /*     printf("\n"); */
            /*     printf("The memory_offset per process\n"); */
            /*     for(int i = 0; i < numtasks; i++) printf("%d ", memory_offset[i]); */
            /* } */
            /* printf("\n"); */


            MPI_Iscatterv(x, size_per_proc, memory_offset, MPI_DOUBLE, new_buffer, size_per_proc[next], MPI_DOUBLE, MASTER, MPI_COMM_WORLD, &req);
        }

        if (i != 0) local_n_old = local_n_curr;
        local_n_curr = size_per_proc[prev]/d;

        if (i != 0) {
            MALLOC(double, ret_old.ndist, MIN(local_n_old, k) * m_per_process);
            MALLOC(uint32_t, ret_old.nidx, MIN(local_n_old, k) * m_per_process);
            memcpy(ret_old.ndist, ret_curr.ndist, sizeof(double) * MIN(local_n_old, k) * m_per_process);
            memcpy(ret_old.nidx, ret_curr.nidx, sizeof(uint32_t) * MIN(local_n_old, k) * m_per_process);
            free(ret_curr.ndist);
            free(ret_curr.nidx);
        }
        ret_curr = kNN(curr_buffer, init_buffer, local_n_curr, m_per_process, d, k);

        /* printf("\nIndeces of kNN ret curr before\n"); */
        /* print_indeces(ret_curr.nidx, ret_curr.m, k); */

        // adjust indeces
        int local_k = MIN(local_n_curr, k);
        for(int j = 0; j < m_per_process; j++) {
            for(int k = 0; k < local_k; k++) {
                ret_curr.nidx[k + j*local_k] += grid; 
            }
        }

        /* printf("\nDistance of kNN ret curr\n"); */
        /* print_dataset_yav(ret_curr.ndist, ret_curr.m, k); */
        /* printf("\nIndeces of kNN ret curr after\n"); */
        /* print_indeces(ret_curr.nidx, ret_curr.m, k); */

        /* if (i != 0) { */
        /*     printf("\nDistance of kNN ret old\n"); */
        /*     print_dataset_yav(ret_old.ndist, ret_old.m, k); */
        /*     printf("\nIndeces of kNN ret old\n"); */
        /*     print_indeces(ret_old.nidx, ret_old.m, k); */
        /* } */

        // comparinggg the ret_curr with the new ret_new
        if (i != 0) {
            // we need a function here. Compare the ndist and nidx
            for (int j = 0; j < m_per_process; j++) {

                double *merged_distance;
                int len_dist1 = MIN(local_n_old, k);
                int len_dist2 = MIN(local_n_curr, k);
                int len_total = len_dist1 + len_dist2;
                MALLOC(double, merged_distance, len_total); 

                memcpy(merged_distance, ret_old.ndist + j*len_dist1, sizeof(double) * len_dist1);
                /* printf("The ret old\n"); */
                /* for(int l = 0; l < len_dist1; l++) printf("%lf ", ret_old.ndist[l + j*len_dist1]); */
                /* printf("\n"); */
                /* printf("The merged distance is\n"); */
                /* for(int l = 0; l < len_dist1; l++) printf("%lf ", merged_distance[l]); */
                /* printf("\n"); */
                memcpy(merged_distance + len_dist1, ret_curr.ndist + j*len_dist2, sizeof(double) * len_dist2);
                /* printf("The merged distance is\n"); */
                /* for(int l = 0; l < len_total; l++) printf("%lf ", merged_distance[l]); */
                /* printf("\n"); */

                uint32_t *merged_indeces;
                MALLOC(uint32_t, merged_indeces, len_total); 

                memcpy(merged_indeces, ret_old.nidx + j*len_dist1, sizeof(uint32_t) * len_dist1);
                /* printf("The merged indeces is\n"); */
                /* for(int l = 0; l < len_dist1; l++) printf("%d ", merged_indeces[l]); */
                /* printf("\n"); */
                memcpy(merged_indeces + len_dist1, ret_curr.nidx + j*len_dist2, sizeof(uint32_t) * len_dist2);
                /* printf("The merged indeces is\n"); */
                /* for(int l = 0; l < len_total; l++) printf("%d ", merged_indeces[l]); */
                /* printf("\n"); */

                if (len_total > k) {
                    qselect(merged_distance, merged_indeces, len_total, k-1);
                    memcpy(ret_per_process.ndist + j*k, merged_distance, sizeof(double) * k);
                    memcpy(ret_per_process.nidx + j*k, merged_indeces, sizeof(uint32_t) * k);
                }
                else {  
                    memcpy(ret_per_process.ndist + j*k, merged_distance, sizeof(double) * len_total);
                    memcpy(ret_per_process.nidx + j*k, merged_indeces, sizeof(uint32_t) * len_total);
                }       

                /* printf("The per process distance is\n"); */
                /* for(int l = 0; l<k; l++) printf("%lf ", ret_per_process.ndist[l + j*k]); */
                /* printf("\n"); */

                /* printf("The per process indeces is\n"); */
                /* for(int l = 0; l<k; l++) printf("%d ", ret_per_process.nidx[l + j*k]); */
                /* printf("\n"); */


                free(merged_distance);
                free(merged_indeces);
            }

            free(ret_old.ndist);
            free(ret_old.nidx);
        }
        else ret_per_process = ret_curr;

        if(numtasks != 1) MPI_Wait(&req, &stat);

        /* if(i <= numtasks-2) { */
        /*     printf("Rank: %d. The next buff is:\n", rank); */
        /*     print_dataset_yav(new_buffer, size_per_proc[next]/d, d); */
        /* } */

    }


    free(new_buffer);
    free(init_buffer);

    printf("\n");

    printf("\nRank %d. Distance of kNN\n", rank);
    print_dataset_yav(ret_per_process.ndist, ret_per_process.m, k);
    printf("\nRank %d. Indeces of kNN\n", rank);
    print_indeces(ret_per_process.nidx, ret_per_process.m, k);


    // Gather to root
    // pass structs to root. How?
    if(rank == MASTER) {
        printf("The n is %d and k is %d\n", n, k);
    }

    //MPI_Gatherv(ret_per_process.ndist, size_per_proc_final[rank], MPI_DOUBLE, ret_master.ndist, size_per_proc_final, memory_offset_final, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    /* MPI_Gatherv(ret_per_process.nidx, size_per_proc_final[rank], MPI_INT, ret_master.nidx, size_per_proc_final, memory_offset_final, MPI_DOUBLE, MASTER, MPI_COMM_WORLD); */

    if (rank == MASTER) {

    /* printf("\nMASTER. Distance of kNN\n"); */
    /* print_dataset_yav(ret_master.ndist, ret_master.m, k); */
    /* printf("\nMASTER. Indeces of kNN\n"); */
    /* print_indeces(ret_master.nidx, ret_master.m, k); */

    }

    free(ret_per_process.ndist);
    free(ret_per_process.nidx);
    if(rank == MASTER) free(x);


    return ret_master;
}

