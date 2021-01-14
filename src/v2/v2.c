#include <stdio.h>
#include <stdint.h>
#include "v0.h"
#include "v1.h"
#include "v2.h"
#include "utils.h"
#include <mpi.h>
#include <math.h>
#include <assert.h>
#include <string.h>

#define LOG2(n) log(n)/log(2)

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
        printf(CYN "==>" RESET " VERSION 2\n");

        if(k > n) {
            printf(RED "Error: " RESET "Number of nearest elements exceeded the number of elements in the corpus set\n");
            errc = -1;
            MPI_Abort(MPI_COMM_WORLD, errc);
        }
        if(numtasks > n) {
            printf(RED "Error: " RESET "Number of processes exceeded the number of elements in the corpus set\n");
            errc = -1;
            MPI_Abort(MPI_COMM_WORLD, errc);
        }
        if(numtasks > n/2) {
            printf(RED "Error: " RESET "Each process should have at least two points\n");
            errc = -1;
            MPI_Abort(MPI_COMM_WORLD, errc);
        }

#ifdef RANDOM
        //srand(time(NULL));
        srand(1);
        MALLOC(double, x, n*d);

        printf("Random generated corpus...\n");
        for (uint32_t i = 0; i< n; i++) {
            for (uint32_t j = 0; j < d; j++) {
                x[i*d + j] = (double)(rand()%100);
            }
        }
#else
        x = read_matrix(&n, &d, argc, argv);
#endif

        ret_master.k = k;
        ret_master.m = n;
        MALLOC(double, ret_master.ndist, n*k);
        MALLOC(uint32_t, ret_master.nidx, n*k);

        //printf("%u\n", k);
        printf("n = %u, d = %u, k = %u\n", n, d, k);
        printf("Corpus array size: %0.3lf MB\n", n*d*8/1e6);
        printf("Total distance matrix size: %0.3lf MB\n", n*n*8/1e6);
    }

#ifndef RANDOM
    MPI_Bcast(&n, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&d, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
#endif

    int corpus_size_per_proc[numtasks]; 
    int distance_offset[numtasks]; 
    memdistr(n, d, numtasks, corpus_size_per_proc, distance_offset); // configure the processes' memory chunks of the nxd corpus set

    double *init_buffer;
    MALLOC(double, init_buffer, corpus_size_per_proc[rank]);
    MPI_Scatterv(x, corpus_size_per_proc, distance_offset, MPI_DOUBLE, init_buffer, corpus_size_per_proc[rank], MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    if(rank == MASTER) free(x);

    int kNN_per_proc[numtasks];
    int kNN_offset[numtasks];
    memdistr(n, k, numtasks, kNN_per_proc, kNN_offset); // configure the processes' memory chunks of the nxk (distance or indeces)

    uint32_t m_per_process = corpus_size_per_proc[rank]/d;

    knnresult ret_per_process;
    ret_per_process.m = m_per_process;
    MALLOC(double, ret_per_process.ndist, m_per_process * k);
    MALLOC(uint32_t, ret_per_process.nidx, m_per_process * k);

    // create a local vp tree per process
    int num_nodes_balanced_per_proc[numtasks];
    int height_tree[numtasks];
    for (int i = 0; i < numtasks; i++) {
        int corpus_points = corpus_size_per_proc[i]/d;
        //if(rank == MASTER) printf("Local corpus points %d\n", corpus_points);
        assert(corpus_points>1);

        // round up n to the next power of 2 finding the nodes of a balanced tree
        float log2points = LOG2(corpus_points);
        height_tree[i] = ceil(log2points); // assume root: 1th height
        //if(height_tree[i] == floor(log2points)) height_tree[i] += 1;
        num_nodes_balanced_per_proc[i] = pow(2, height_tree[i]) - 1;
        //if(rank == MASTER) printf("The height of the complete tree is %d (assume root is in the 1st height)\n", height_tree[i]);
        //if(rank == MASTER) printf("The total number of nodes for a balanced tree %d\n", num_nodes_balanced_per_proc[i]);
    }

    uint32_t *indeces_per_process;
    MALLOC(uint32_t, indeces_per_process, m_per_process);
    /* printf("\n"); */
    for(uint32_t i = 0; i < m_per_process; i++) {
        indeces_per_process[i] = i;
    }
    int offset = distance_offset[rank]/d;
    adjust_indeces(indeces_per_process, 1, m_per_process, offset);

    double *vpt_buffer;
    MALLOC(double, vpt_buffer, corpus_size_per_proc[rank]);
    memcpy(vpt_buffer, init_buffer, sizeof(double) * corpus_size_per_proc[rank]);

    const float target_height_tree_percent = 1;
    printf("The height percentage is %f\n", target_height_tree_percent);

    Vptree vpt(vpt_buffer, indeces_per_process, m_per_process, d, num_nodes_balanced_per_proc[rank], height_tree[rank], target_height_tree_percent);

    free(vpt_buffer);
    free(indeces_per_process);


    for (int i = 0; i < m_per_process * k; i++) {
        ret_per_process.ndist[i] = INFINITY;
        ret_per_process.nidx[i] = UINT32_MAX;
    }

    knnresult ret_curr;

    double *send_vp_mu;
    double *send_vp_coords;
    int *send_vp_index;

    double *recv_vp_mu;
    double *recv_vp_coords;
    int *recv_vp_index;

    MPI_Request reqs[6];
    MPI_Status stats[6];
    int tags[3];
    tags[0] = 0;
    tags[1] = 1;
    tags[2] = 2;

    TIC()

    for(int i = 0; i < numtasks; i++) {
        int isSend = 0;

        int offset = distance_offset[rank]/d;
        int local_n = corpus_size_per_proc[rank]/d;
        int local_num_nodes_balanced = num_nodes_balanced_per_proc[rank];

        if(i != 0) {
            MALLOC(double, vpt.vp_mu, local_num_nodes_balanced);
            MALLOC(double, vpt.vp_coords, local_num_nodes_balanced * d);
            MALLOC(int, vpt.vp_index, local_num_nodes_balanced);

            // copy receive to current
            memcpy(vpt.vp_mu, recv_vp_mu, sizeof(double) * local_num_nodes_balanced);
            memcpy(vpt.vp_coords, recv_vp_coords, sizeof(double) * local_num_nodes_balanced * d);
            memcpy(vpt.vp_index, recv_vp_index, sizeof(int) * local_num_nodes_balanced);

            free(recv_vp_mu);
            free(recv_vp_coords);
            free(recv_vp_index);

            // init for search tree skipping the creation
            vpt.init_before_search(local_n, local_num_nodes_balanced, height_tree[rank], target_height_tree_percent, vpt.vp_mu, vpt.vp_coords, vpt.vp_index);
        }

        // one step ahead. Calculate the next buffer to avoid communication cost
        if (i <= numtasks-2) {
            MALLOC(double, send_vp_mu, local_num_nodes_balanced);
            MALLOC(double, send_vp_coords, local_num_nodes_balanced * d);
            MALLOC(int, send_vp_index, local_num_nodes_balanced);

            // copy init to send
            memcpy(send_vp_mu, vpt.vp_mu, sizeof(double) * local_num_nodes_balanced);
            memcpy(send_vp_coords, vpt.vp_coords, sizeof(double) * local_num_nodes_balanced * d);
            memcpy(send_vp_index, vpt.vp_index, sizeof(int) * local_num_nodes_balanced);

            int prev = rank-1;
            int next = rank+1;
            if (rank == 0)  prev = numtasks - 1;
            if (rank == (numtasks - 1))  next = 0;            

            MPI_Isend(send_vp_mu, local_num_nodes_balanced, MPI_DOUBLE, prev, tags[0], MPI_COMM_WORLD, &reqs[0]);
            MPI_Isend(send_vp_coords, local_num_nodes_balanced * d, MPI_DOUBLE, prev, tags[1], MPI_COMM_WORLD, &reqs[1]);
            MPI_Isend(send_vp_index, local_num_nodes_balanced, MPI_INT, prev, tags[2], MPI_COMM_WORLD, &reqs[2]);

            rotate_left(corpus_size_per_proc, numtasks);
            rotate_left(num_nodes_balanced_per_proc, numtasks);
            rotate_left(height_tree, numtasks);

            local_num_nodes_balanced = num_nodes_balanced_per_proc[rank];

            MALLOC(double, recv_vp_mu, local_num_nodes_balanced);
            MALLOC(double, recv_vp_coords, local_num_nodes_balanced * d);
            MALLOC(int, recv_vp_index, local_num_nodes_balanced);

            MPI_Irecv(recv_vp_mu, local_num_nodes_balanced, MPI_DOUBLE, next, tags[0], MPI_COMM_WORLD, &reqs[3]);
            MPI_Irecv(recv_vp_coords, local_num_nodes_balanced * d, MPI_DOUBLE, next, tags[1], MPI_COMM_WORLD, &reqs[4]);
            MPI_Irecv(recv_vp_index, local_num_nodes_balanced, MPI_INT, next, tags[2], MPI_COMM_WORLD, &reqs[5]);

            isSend = 1;
        }

        ret_curr = kNN_vptree(vpt, init_buffer, local_n, m_per_process, d, k);
        free(vpt.vp_mu);
        free(vpt.vp_coords);
        free(vpt.vp_index);

        // compare the old data with the newest data
        if (i != 0) {
            for (int j = 0; j < m_per_process; j++) {

                int len_curr = MIN(local_n, k);
                int len_total = k + len_curr;

                double *merged_distance;
                MALLOC(double, merged_distance, len_total); 

                memcpy(merged_distance, ret_per_process.ndist + j*k, sizeof(double) * k);
                memcpy(merged_distance + k, ret_curr.ndist + j*len_curr, sizeof(double) * len_curr);

                uint32_t *merged_indeces;
                MALLOC(uint32_t, merged_indeces, len_total); 

                memcpy(merged_indeces, ret_per_process.nidx + j*k, sizeof(uint32_t) * k);
                memcpy(merged_indeces + k, ret_curr.nidx + j*len_curr, sizeof(uint32_t) * len_curr);

                // update
                qselect_and_indeces(merged_distance, merged_indeces, len_total, k-1);
                memcpy(ret_per_process.ndist + j*k, merged_distance, sizeof(double) * k);
                memcpy(ret_per_process.nidx + j*k, merged_indeces, sizeof(uint32_t) * k);

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
            free(ret_curr.nidx);
        }

        if(isSend) {
            //TIC()
            MPI_Waitall(6, reqs, stats);
            /* printf("Freeing sending buffer...\n"); */
            free(send_vp_mu);
            free(send_vp_coords);
            free(send_vp_index);
            //TOC(RED "Cost " RESET "for syncing: %lf\n")
        }

    }

    free(init_buffer);
    clock_gettime(CLOCK_MONOTONIC, &toc);

    //printf(CYN "Rank: %d. " RESET "Time elapsed calculating per process kNN (seconds): %lf\n", rank, diff_time(tic, toc));

    printf("Total number of nodes visited %d of avaialbe %d\n", vpt.total_nodes_visited, m_per_process*n);

    //if(rank == MASTER) {TIC();}

    // Gather to root
    MPI_Gatherv(ret_per_process.ndist, kNN_per_proc[rank], MPI_DOUBLE, ret_master.ndist, kNN_per_proc, kNN_offset, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Gatherv(ret_per_process.nidx, kNN_per_proc[rank], MPI_INT, ret_master.nidx, kNN_per_proc, kNN_offset, MPI_INT, MASTER, MPI_COMM_WORLD);


    //if(rank == MASTER) {TOC(RED "\nCOST " RESET "Gatherv: %lf\n")}

    free(ret_per_process.ndist);
    free(ret_per_process.nidx);

    return ret_master;
}
