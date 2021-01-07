#include <time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
#include "mmio.h"
#include "utils.h"

double diff_time (struct timespec start, struct timespec end) {
    uint32_t diff_sec = (end.tv_sec - start.tv_sec);
    int32_t diff_nsec = (end.tv_nsec - start.tv_nsec);
    if ((end.tv_nsec - start.tv_nsec) < 0) {
        diff_sec -= 1;
        diff_nsec = 1e9 + end.tv_nsec - start.tv_nsec;
    }

    return (1e9*diff_sec + diff_nsec)/1e9;
}

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


void print_input_file(FILE *f, double *array, uint32_t row, uint32_t col) {
    for (uint32_t i = 0; i < row; i++) {
        for (uint32_t j = 0; j < col; j++) {
            fprintf(f, "%lf ", array[i*col + j]);
        }
        fprintf(f, "\n");
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
    for(uint32_t i = 0; i < row; i++)
    {
        if(i == 0)
            printf("[[");
        else
            printf(" [");
        for(uint32_t j = 0; j < col; j++) {
            if (j != col -1) printf("%d,", array[i*col + j]);
            else printf("%d", array[i*col + j]);
        }
        if(i == row -1)
            printf("]]\n");
        else
            printf("],\n");
    }
}

void print_output_file(FILE *f, double *dist, uint32_t *indeces, uint32_t row, uint32_t col) {
    for(uint32_t i = 0; i < row; i++)
    {
        for(uint32_t j = 0; j < col; j++) fprintf(f, "%lf ", dist[i*col + j]);

            fprintf(f, "\n");
    }

    for(uint32_t i = 0; i < row; i++)
    {
        for(uint32_t j = 0; j < col; j++) fprintf(f, "%u ", indeces[i*col + j]);

        fprintf(f, "\n");
    }
}


// D = sqrt(sum(X.^2,2) - 2 * X*Y.' + sum(Y.^2,2).');
double *euclidean_distance(double *x, double *y, uint32_t n, uint32_t d, uint32_t m) {
    printf("Calculating distance matrix nxm approach\n");
    struct timespec tic;
    struct timespec toc;

    //TIC()

    // d1 = sum(X.^2,2)
    double *d1;
    MALLOC(double, d1, n);
#if CBLAS_DDOT == 0
    for (uint32_t i = 0; i < n; i++) {
        d1[i] = 0;
        for (uint32_t j = 0; j < d; j++) {
            d1[i] += x[i*d + j] * x[i*d + j]; 
        }
    }
#elif CBLAS_DDOT == 1
    for (uint32_t i = 0; i < n; i++) d1[i] = cblas_ddot(d, x + i*d, 1, x + i*d, 1);
#endif

    // d2 = sum(Y.^2,2)
    double *d2;
    MALLOC(double, d2, m);
#if CBLAS_DDOT == 0
    for (uint32_t i = 0; i < m; i++) {
        d2[i] = 0;
        for (uint32_t j = 0; j < d; j++) {
            d2[i] += y[i*d + j] * y[i*d + j]; 
        }
    }
#elif CBLAS_DDOT == 1
    for (uint32_t i = 0; i < m; i++) d2[i] = cblas_ddot(d, y + i*d, 1, y + i*d, 1);
#endif

    //TOC("Time elapsed calculating dot product (seconds): %lf\n")

    // D = -2 * X * Y.'
    double *distance;
    MALLOC(double, distance, n*m);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, d, -2, x, d, y, d, 0, distance, m);

    // D = sqrt(D + d1 + d2.')
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < m; j++) {
            distance[i*m + j] += d1[i] + d2[j];
            distance[i*m + j] = sqrt(distance[i*m + j]);
        }
    }

    free(d1);
    free(d2);

    return distance;
}

// naive euclidean distance matrix. That way we don't need to transpose the matrix though
double *euclidean_distance_naive(double *x, double *y, uint32_t n, uint32_t d, uint32_t m) {
    printf("Calculating distance matrix naive approach\n");
    double *distance;
    MALLOC(double, distance, n*m);

    for(uint32_t i = 0; i < m; i++) {
        for(uint32_t j = 0; j < n; j++) {
            distance[n*i + j] = 0;
            for (uint32_t k = 0; k < d; k++) {
                distance[n*i + j] += (x[j*d + k] - y[i*d + k]) * (x[j*d + k] - y[i*d + k]);
            }
            distance[n*i + j] = sqrt(distance[n*i + j]);
        }
    }

    return distance;
}


// D' = sqrt(sum(X.^2,2).' - 2 * Y * X.' + sum(Y.^2, 2))
double *euclidean_distance_notrans(double *x, double *y, uint32_t n, uint32_t d, uint32_t m) {
    printf("Calculating distance matrix mxn approach [%u, %u]\n", m, n);
    struct timespec tic;
    struct timespec toc;

    //TIC()

    // d1 = sum(X.^2,2)
    double *d1;
    MALLOC(double, d1, n);
#if CBLAS_DDOT == 0
    for (uint32_t i = 0; i < n; i++) {
        d1[i] = 0;
        for (uint32_t j = 0; j < d; j++) {
            d1[i] += x[i*d + j] * x[i*d + j]; 
        }
    }
#elif CBLAS_DDOT == 1
    for (uint32_t i = 0; i < n; i++) d1[i] = cblas_ddot(d, x + i*d, 1, x + i*d, 1);
#endif

    // d2 = sum(Y.^2,2)
    double *d2;
    MALLOC(double, d2, m);
#if CBLAS_DDOT == 0
    for (uint32_t i = 0; i < m; i++) {
        d2[i] = 0;
        for (uint32_t j = 0; j < d; j++) {
            d2[i] += y[i*d + j] * y[i*d + j]; 
        }
    }
#elif CBLAS_DDOT == 1
    for (uint32_t i = 0; i < m; i++) d2[i] = cblas_ddot(d, y + i*d, 1, y + i*d, 1);
#endif

    //TOC("Time elapsed calculating dot product (seconds): %lf\n")

    // D = -2 * Y * X.'
    double *distance;
    MALLOC(double, distance, n*m);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, d, -2, y, d, x, d, 0, distance, n);

    // D = sqrt(D + d1.' + d2)
    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < n; j++) {
            distance[i*n + j] += d1[j] + d2[i];
            distance[i*n + j] = sqrt(distance[i*n + j]);
        }
    }

    free(d1);
    free(d2);

    return distance;
}

void transpose(double *src, double *dst, const uint32_t N, const uint32_t M) {
    //#pragma omp parallel for
    for(uint32_t n = 0; n<N*M; n++) {
        uint32_t i = n/N;
        uint32_t j = n%N;
        dst[n] = src[M*j + i];
    }
}

uint32_t partition(double arr[], uint32_t start, uint32_t end) { 
    int x = arr[end],
    i = start; 
    for (uint32_t j = start; j <= end - 1; j++) { 
        if (arr[j] <= x) { 
            //swap(arr[i], arr[j]); 
            auto temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
            i++; 
        } 
    } 
    //swap(arr[i], arr[end]); 
    auto temp = arr[i];
    arr[i] = arr[end];
    arr[end] = temp;

    return i; 
} 

int64_t quickselect(double arr[], int64_t start, int64_t end, int64_t k) { 
    if (k > 0 && k <= end - start + 1) { 
  
        int32_t index = partition(arr, start, end); 
  
        if (index - start == k - 1) 
            return index; 
  
        if (index - start > k - 1)  
            return quickselect(arr, start, index - 1, k); 
  
        return quickselect(arr, index + 1, end, k - index + start - 1); 
    } 
  
    return -1; 
} 


double qselect_and_indeces(double *v, uint32_t *idx, int64_t len, int64_t k) {
#define SWAPval(a, b) { tmp1 = v[a]; v[a] = v[b]; v[b] = tmp1; }
#define SWAPidx(a, b) { tmp2 = idx[a]; idx[a] = idx[b]; idx[b] = tmp2; }

	int32_t i, st;
    double tmp1;
    double tmp2;
    double tmp3;
 
    /* int median = len - 1; */
    /* if(len>3) { */
    /*     median = medianThree(v, 0, len/2, len-1); */
    /*     SWAPval(median, len-1); */
    /*     SWAPidx(median, len-1); */
    /* } */

	for (st = i = 0; i < len - 1; i++) {
		if (v[i] > v[len-1]) continue;
		SWAPval(i, st);
		SWAPidx(i, st);
		st++;
	}
 
	SWAPval(len-1, st);
	SWAPidx(len-1, st);
 
	return k == st	?v[st] 
			:st > k	? qselect_and_indeces(v, idx, st, k)
				: qselect_and_indeces(v + st, idx + st, len - st, k - st);
}

double qselect(double *v,int64_t len, int64_t k) {
#define SWAPval(a, b) { tmp1 = v[a]; v[a] = v[b]; v[b] = tmp1; }

	int32_t i, st;
    double tmp1;
 
    /* int median = len - 1; */
    /* if(len>3) { */
    /*     median = medianThree(v, 0, len/2, len-1); */
    /*     SWAPval(median, len-1); */
    /*     SWAPidx(median, len-1); */
    /* } */

	for (st = i = 0; i < len - 1; i++) {
		if (v[i] > v[len-1]) continue;
		SWAPval(i, st);
		st++;
	}
 
	SWAPval(len-1, st);
 
	return k == st	?v[st] 
			:st > k	? qselect(v,st, k)
				: qselect(v + st, len - st, k - st);
}

int medianThree(double *array, int a, int b, int c) {
    if ((array[a] > array[b]) != (array[a] > array[c])) 
        return a;
    else if ((array[b] > array[a]) != (array[b] > array[c])) 
        return b;
    else
        return c;
}

void memdistr(uint32_t n, uint32_t d, int numtasks, int *size_per_proc, int *memory_offset) {
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
}

void rotate_left(int *arr, int size) {
    int temp = arr[0];
    for(int i = 0; i < size-1; i++) arr[i] = arr[i+1];
    arr[size-1] = temp;
}

void adjust_indeces(uint32_t *arr, uint32_t rows, uint32_t cols, int offset) {
    for(uint32_t i = 0; i < rows; i++) {
        for(uint32_t j = 0; j < cols; j++) {
            arr[j + i*cols] += offset; 
        }
    }
}
