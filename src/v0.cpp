#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "include/main.h"

// with ddot 1, we have a slight better performance, but they are pretty much the same
// todo: check the difference between cblas_dgemm and manual matrix multiplication
#define DDOT 0

#define CBLAS_TRANS 1

// CPP 1, way slower...
#define CPP 0

#if CPP == 1
#include <algorithm> 
#include <unordered_map>
#endif

extern void print_dataset(double *array, uint32_t row, uint32_t col);
extern void print_dataset_yav(double *array, uint32_t row, uint32_t col);
extern void print_indeces(uint32_t *array, uint32_t row, uint32_t col);

void transpose(double *src, double *dst, const uint32_t N, const uint32_t M);
uint32_t partition(double arr[], uint32_t start, uint32_t end);
int64_t quickselect(double arr[], int64_t start, int64_t end, int64_t k);
double qselect(double *v, uint32_t *idx, int64_t len, int64_t k);

knnresult kNN(double *x, double *y, uint32_t n, uint32_t m, uint32_t d, uint32_t k) {

    knnresult ret;
    ret.ndist = (double*)malloc(m * k * sizeof(double));
    ret.nidx  = (uint32_t*)malloc(m * k * sizeof(uint32_t));
    ret.k = k; // included the query to be tested?
    ret.m = m;

    struct timespec tic;
    struct timespec toc;
    // D = sqrt(sum(X.^2,2) - 2 * X*Y.' + sum(Y.^2,2).');
    
    // d1 = sum(X.^2,2)
    clock_gettime(CLOCK_MONOTONIC, &tic);
    double d1[n];
#if DDOT == 0
    for (uint32_t i = 0; i < n; i++) {
        d1[i] = 0;
        for (uint32_t j = 0; j < d; j++) {
            d1[i] += x[i*d + j] * x[i*d + j]; 
        }
    }
#elif DDOT == 1
    for (uint32_t i = 0; i < n; i++) d1[i] = cblas_ddot(d, x + i*d, 1, x + i*d, 1);
#endif

    // d2 = sum(Y.^2,2)
    double d2[m];
#if DDOT == 0
    for (uint32_t i = 0; i < m; i++) {
        d2[i] = 0;
        for (uint32_t j = 0; j < d; j++) {
            d2[i] += y[i*d + j] * y[i*d + j]; 
        }
    }
#elif DDOT == 1
    for (uint32_t i = 0; i < m; i++) d2[i] = cblas_ddot(d, y + i*d, 1, y + i*d, 1);
#endif

    clock_gettime(CLOCK_MONOTONIC, &toc);
    printf("Time elapsed calculating dot product (seconds): %lf\n", diff_time(tic, toc));


    clock_gettime(CLOCK_MONOTONIC, &tic);
    // D = -2 * X * Y.'
    double *distance = (double*)malloc(n * m * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, d, -2, x, d, y, d, 0, distance, m);

    // D = sqrt(D + d1 + d2)
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < m; j++) {
            distance[i*m + j] += d1[i] + d2[j];
            distance[i*m + j] = sqrt(distance[i*m + j]);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &toc);
    printf("Time elapsed calculating distance matrix (seconds): %lf\n", diff_time(tic, toc));

    /* printf("\nThe distance matrix m x n: \n"); */
    /* print_dataset_yav(distance, n, m); */

    // performance issue (cache) 
    clock_gettime(CLOCK_MONOTONIC, &tic);

    double *distance_t = (double*)malloc(m * n * sizeof(double)); // mxn
#if CBLAS_TRANS == 0
    transpose(distance, distance_t, n, m);
#elif CBLAS_TRANS == 1
    cblas_domatcopy(CblasRowMajor, CblasTrans, n, m, 1, distance, m, distance_t, n); 
#endif

    free(distance);

    clock_gettime(CLOCK_MONOTONIC, &toc);
    printf("Time elapsed calculating transpose distance mxn matrix (seconds): %lf\n", diff_time(tic, toc));

    /* printf("\nThe transpose distance matrix m x n: \n"); */
    /* print_dataset_yav(distance_t, m, n); */

    /* we need to be compatible with elearning tester */
    int isFirst = 1; // indexes 1 or 0 based?
    printf("Sorting distance matrix until %uth element...\n", k);
    clock_gettime(CLOCK_MONOTONIC, &tic);
    for(uint32_t i = 0; i < m; i++) {
        double *mth_array = distance_t + i*n; 

#if CPP == 1

        // map each distance value to the index of the corpus
        std::unordered_map<double, uint32_t> mth_map; 
        for (uint32_t j=0; j < n; j++) mth_map[mth_array[j]] = j; // costs time

        // find the k smallest
        std::nth_element(mth_array, mth_array + k-1, mth_array + n);
        //memcpy(ret.ndist + i*k, mth_array, sizeof(double)*k);

        for(uint32_t j = 0; j < k; j++) {
            ret.ndist[i*k + j] = mth_array[j]; 

            //ret.nidx[i*k + j] = mth_map[mth_array[j]] + isFirst;
            
            // in case there are duplicate distance values
            auto range = mth_map.equal_range(mth_array[j]); 
            auto count = mth_map.count(mth_array[j]);
            for (auto it = range.first; it != range.second; it++) {
                ret.nidx[i*k + j] = it->second + isFirst;
                if(count-- != 1) {
                    if (j < k) j++; 
                    else break;
                }
            }
        } 
#elif CPP == 0

        uint32_t *mth_indeces = (uint32_t*)malloc(n * sizeof(uint32_t));
        for(uint32_t j = 0; j < n; j ++) mth_indeces[j] = j + isFirst;

        qselect(mth_array, mth_indeces, n, k-1);
        memcpy(ret.ndist + i*k, mth_array, sizeof(double) * k);
        memcpy(ret.nidx + i*k, mth_indeces, sizeof(uint32_t) * k);

        free(mth_indeces);

#endif
    }

    free(distance_t);

    clock_gettime(CLOCK_MONOTONIC, &toc);
    printf("Time elapsed calculating kNN given distance matrix (seconds): %lf\n", diff_time(tic, toc));

    return ret;
}

void transpose(double *src, double *dst, const uint32_t N, const uint32_t M) {
    //#pragma omp parallel for
    for(uint32_t n = 0; n<N*M; n++) {
        uint32_t i = n/N;
        uint32_t j = n%N;
        dst[n] = src[M*j + i];
    }
}

// source gfg
// fix me: median of 3
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

// source gfg
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


// rosetta code wiki modified
double qselect(double *v, uint32_t *idx, int64_t len, int64_t k) {
#	define SWAPval(a, b) { tmp1 = v[a]; v[a] = v[b]; v[b] = tmp1; }
#	define SWAPidx(a, b) { tmp2 = idx[a]; idx[a] = idx[b]; idx[b] = tmp2; }
	int32_t i, st;
    double tmp1;
    double tmp2;
 
	for (st = i = 0; i < len - 1; i++) {
		if (v[i] > v[len-1]) continue;
		SWAPval(i, st);
		SWAPidx(i, st);
		st++;
	}
 
	SWAPval(len-1, st);
	SWAPidx(len-1, st);
 
	return k == st	?v[st] 
			:st > k	? qselect(v, idx, st, k)
				: qselect(v + st, idx + st, len - st, k - st);
}
