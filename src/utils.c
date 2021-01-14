#include <time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
#include "mmio.h"
#include "utils.h"
#include <string.h>
#include <time.h>

double diff_time (struct timespec start, struct timespec end) {
    uint32_t diff_sec = (end.tv_sec - start.tv_sec);
    int32_t diff_nsec = (end.tv_nsec - start.tv_nsec);
    if ((end.tv_nsec - start.tv_nsec) < 0) {
        diff_sec -= 1;
        diff_nsec = 1e9 + end.tv_nsec - start.tv_nsec;
    }

    return (1e9*diff_sec + diff_nsec)/1e9;
}


double *read_matrix(uint32_t *n, uint32_t *d, int argc, char *argv[]) {
    FILE *f;
    double *x;

    if(argc < 2) {
        printf("Missed command line arguements\n");
		fprintf(stderr, "Usage: %s [matrix-filename]\n", argv[0]);
		exit(1);
	}
    else {
        if((f = fopen(argv[1], "r")) == NULL) { 
            printf("Can't open file\n");
            exit(1);  
        }
    }

    if(strstr(argv[1], "corel")!=NULL)              x = read_corel(f, argv[1], n, d);
    else if(strstr(argv[1], "features")!=NULL)      x = read_features(f, n, d);
    else if(strstr(argv[1], "MiniBooNE_PID")!=NULL) x = read_mini(f, n, d);
    else if(strstr(argv[1], "tv")!=NULL)            x = read_tv(f, argv[1], n, d);

    fclose(f);

    return x;
}

double *read_corel(FILE *f, char *file_name, uint32_t *n, uint32_t *d) {
    uint32_t rows = 0;
    uint32_t cols = 0;

    rows = 68040;
    if(strcmp(file_name,      "datasets/corel/ColorMoments.asc")==0)   cols = 9;
    else if(strcmp(file_name, "datasets/corel/CoocTexture.asc")==0)    cols = 16;
    else if(strcmp(file_name, "datasets/corel/ColorHistogram.asc")==0) cols = 32;
    else {printf("Not supported corel matrix\nPlease use ColorHistogram or CoocTexture or ColorMoments\n"); exit(-1);}
        
    double *x;
    MALLOC(double, x, rows * cols);

    for(uint32_t i = 0; i < rows; i++) {
        // skip first value
        int skip;
        fscanf(f, "%d", &skip);
        for(uint32_t j = 0; j < cols; j++) {
            if(fscanf(f, "%lf", &x[i*cols + j]) != 1) exit(-1);
        }
    }

    *d = cols;
    *n = rows;

    return x;

}

double *read_features(FILE *f, uint32_t *n, uint32_t *d) {
    uint32_t rows = 0;
    uint32_t cols = 0;

    char *line;
    MALLOC(char, line, 1024*1024);

    for(int skip=0;skip<4;skip++){
        fscanf(f,"%s\n", line);
    }
    free(line);

    rows = 106574;
    cols = 518;

    double *x;
    MALLOC(double, x, rows * cols);

    for(uint32_t i = 0; i < rows; i++) {
        // skip first value
        int skip;
        fscanf(f, "%d,", &skip);
        for(uint32_t j = 0; j < cols; j++) {
            if(fscanf(f, "%lf,", &x[i*cols + j]) != 1) exit(-1);
        }
    }

    *d = cols;
    *n = rows;

    return x;
}

double *read_mini(FILE *f, uint32_t *n, uint32_t *d) {
    uint32_t rows = 130064;
    uint32_t cols = 50;

    for(int i=0, skip = 0; i<2; i++){
        fscanf(f, "%d ", &skip);
    }

    double *x;
    MALLOC(double, x, rows * cols);

    for(uint32_t i = 0; i < rows; i++) {
        // skip first value
        int skip;
        fscanf(f, "%d ", &skip);
        for(uint32_t j = 0; j < cols; j++) {
            if(fscanf(f, "%lf ", &x[i*cols + j]) != 1) exit(-1);
        }
    }

    *d = cols;
    *n = rows;

    return x;
}

double *read_tv(FILE *f, char *file_name, uint32_t *n, uint32_t *d){
    uint32_t rows = 0;
    uint32_t cols = 0;

    if(strcmp(file_name,      "datasets/tv/BBC.txt")==0)      rows = 17720;
    else if(strcmp(file_name, "datasets/tv/CNN.txt")==0)      rows = 22545;
    else if(strcmp(file_name, "datasets/tv/NDTV.txt")==0)     rows = 17051;
    else if(strcmp(file_name, "datasets/tv/CNNIBN.txt")==0)   rows = 3317;
    else if(strcmp(file_name, "datasets/tv/TIMESNOW.txt")==0) rows = 39252;
    else printf("Not supported tv matrix\n Please check the filename\n");

    cols = 17;
        
    double *x;
    MALLOC(double, x, rows * cols);

    for(uint32_t i = 0; i < rows; i++) {
        // skip first value
        int skip;
        fscanf(f, "%d ", &skip);
        for(uint32_t j = 0; j < cols; j++) {
            if(fscanf(f, "%d:%lf ", &skip, &x[i*cols + j]) != 2) exit(-1);
        }
        fscanf(f,"%*[^\n]\n");
    }

    *d = cols;
    *n = rows;

    return x;
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

/*
 * Calculate dot product using cblas routine
 *
 * 0 --> false
 * 1 --> true 
 */
#define CBLAS_DDOT 1
// D' = sqrt(sum(X.^2,2).' - 2 * Y * X.' + sum(Y.^2, 2))
double *euclidean_distance(double *x, double *y, uint32_t n, uint32_t d, uint32_t m) {
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

double qselect_and_indeces(double *v, uint32_t *idx, int64_t len, int64_t k) {
#define SWAPval(a, b) { tmp1 = v[a]; v[a] = v[b]; v[b] = tmp1; }
#define SWAPidx(a, b) { tmp2 = idx[a]; idx[a] = idx[b]; idx[b] = tmp2; }

	int32_t i, st;
    double tmp1;
    double tmp2;
 
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

void memdistr(int n, int d, int numtasks, int *size_per_proc, int *memory_offset) {
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

