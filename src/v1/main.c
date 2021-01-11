#include <cblas.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <stdlib.h>
#include "v0.h"
#include "v1.h"
#include "utils.h"
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

    uint32_t n = (uint32_t)10;
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

        printf("\nDistance of kNN\n");
        print_dataset_yav(ret.ndist, ret.m, k);
        printf("\nIndeces of kNN\n");
        print_indeces(ret.nidx, ret.m, k);

        log = fopen("logs/v1_log.txt", "w");

        // for comparing v1 and v2 sort the data
        for(int j = 0; j < k; j++) {
            for(int i = 0; i < ret.m; i++) {
                qselect_and_indeces(ret.ndist + i*k, ret.nidx + i*k, k, j);
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
