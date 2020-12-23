#include "main.h"
#include "mmio.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>


void mm2coo(int argc, char *argv[], uint32_t **rows, uint32_t **columns, uint32_t nnz, uint32_t n) {
    MM_typecode matcode;
    FILE *f;
    uint32_t r, c; // MxN dimensions (square matrix M=N) 
    // double *val; // dont need this. Our matrices are binary 1 or zero

    // expecting a filename to read (./main <filename>)
    if (argc < 2) {
        printf("Missed command line arguements\n");
		fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
		exit(1);
	}
    else {
        if ((f = fopen(argv[1], "r")) == NULL) { 
            printf("Can't open file\n");
            exit(1);  
        }
    }

    if (mm_read_banner(f, &matcode) != 0) {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }

    // what MM formats do you support?
    if (!(mm_is_matrix(matcode) && mm_is_coordinate(matcode) && mm_is_pattern(matcode) && 
            mm_is_symmetric(matcode))) {
        printf("Sorry, this application does not support ");
        printf("Matrix Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    /* find out size of sparse matrix .... */
    if ((mm_read_mtx_crd_size(f, &r, &c, &nnz)) !=0) exit(1);
    //printf("Number of nnz: %u\n", nnz);
    n = r;
    //printf("Rows/columns: %u\n", n);

    *rows = (uint32_t*) malloc(nnz * sizeof(uint32_t));
    *columns = (uint32_t*) malloc(nnz * sizeof(uint32_t));

    uint32_t x,y = 0;
    for (uint32_t i=0; i<nnz; i++) {
        fscanf(f, "%u %u\n", &x, &y);
        if (x == y) {
            nnz--;
            i--;
            continue;
        }
        (*rows)[i] = x;
        (*columns)[i] = y;
        (*rows)[i]--;  /* adjust from 1-based to 0-based */
        (*columns)[i]--;

        //printf("Elements: [%lu, %lu]\n", rows[i], columns[i]);
    }

    printf("Success, MM format is converted to COO\n");

    if (f !=stdin) {
        fclose(f);
        //printf("File is successfully closed\n");
    }
}

