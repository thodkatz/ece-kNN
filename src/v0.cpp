#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "include/main.h"


/*
 * Calculate dot product using cblas routine
 *
 * 0 --> false
 * 1 --> true 
 */
#define CBLAS_DDOT 0

/*
 * Calculate transpose matrix using cblas routine
 *
 * 0 --> false
 * 1 --> true
 */
#define CBLAS_TRANS 1

/*
 * Create a transpose of distance matrix (nxm)
 *
 * Note: The data is stored in row major format.
 * For the knn, given the distance matrix (nxm) we need
 * to iterate each column for the k-select that will 
 * result to cache miss.
 *
 * Is it worth it to create first a transpose matrix?
 *
 * 0 --> don't create
 * 1 --> create
 *
 * Results: Much better performance to transpose first
 */
#define TRANS 0

/*
 * Use STL algorithms for k-select
 * 0 --> false
 * 1 --> true
 */
#define CPP 0

/*
 * Calculate euclidean distance matrix
 *
 *
 * 0 --> naive d^2 = (x_i - y_i)^2 Requires TRANS == 0
 * 1 --> D = sqrt(sum(X.^2,2) - 2 * X*Y.' + sum(Y.^2,2).') Requires TRANS == 1
 * 2 --> D' = sqrt(sum(X.^2,2).' - 2 * Y * X.' + sum(Y.^2, 2)) (mxn) Requires TRANS == 0
 */
#define MATRIX 2

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
double *euclidean_distance(double *x, double *y, uint32_t n, uint32_t d, uint32_t m);
double *euclidean_distance_naive(double *x, double *y, uint32_t n, uint32_t d, uint32_t m);
double *euclidean_distance_notrans(double *x, double *y, uint32_t n, uint32_t d, uint32_t m);

knnresult kNN(double *x, double *y, uint32_t n, uint32_t m, uint32_t d, uint32_t k) {

    knnresult ret;
    MALLOC(double, ret.ndist, m*k);
    MALLOC(uint32_t, ret.nidx, m*k);
    ret.k = k; // included the query to be tested?
    ret.m = m;

    struct timespec tic;
    struct timespec toc;
    
    TIC();
#if MATRIX == 0
    double *distance = euclidean_distance_naive(x, y, n, d, m);
#elif MATRIX == 1
    double *distance = euclidean_distance(x, y, n, d, m);
#elif MATRIX == 2
    double *distance = euclidean_distance_notrans(x, y, n, d, m);
#endif
    TOC("Time elapsed calculating distance matrix (seconds): %lf\n");

#if MATRIX == 1
    /* printf("\nThe distance matrix n x m: \n"); */
    /* print_dataset_yav(distance, n, m); */
#elif MATRIX == 2
    /* printf("\nThe distance matrix m x n: \n"); */
    /* print_dataset_yav(distance, m, n); */
#endif


#if TRANS == 1
    double *distance_trans;
    if (n != m) {
        TIC();

        // create transpose to exploit cache hits 
        MALLOC(double, distance_trans, m*n);
#if CBLAS_TRANS == 0  
        transpose(distance, distance_trans, n, m);
#elif CBLAS_TRANS == 1 
        cblas_domatcopy(CblasRowMajor, CblasTrans, n, m, 1, distance, m, distance_trans, n); 
#endif

        free(distance);

        TOC("Time elapsed calculating transpose distance mxn matrix (seconds): %lf\n");

        /* printf("\nThe transpose distance matrix m x n: \n"); */
        /* print_dataset_yav(distance_trans, m, n); */
    }
    else distance_trans = distance; // symmetric
#endif

    int isFirst = 1; // indeces 1 or 0 based?
    printf("Sorting distance matrix until %uth element...\n", k);

    TIC();

    for(uint32_t i = 0; i < m; i++) {
#if TRANS == 1
        double *mth_array = distance_trans + i*n; 
#elif TRANS == 0
        double *mth_array; 
#if MATRIX == 1
        MALLOC(double, mth_array, n);
        for(uint32_t j = 0; j < n; j++) mth_array[j] = distance[i + j*m];
#elif MATRIX == 0 || MATRIX == 2
        mth_array = distance + i*n;
#endif
#endif

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
        uint32_t *mth_indeces;
        MALLOC(uint32_t, mth_indeces, n);
        for(uint32_t j = 0; j < n; j ++) mth_indeces[j] = j + isFirst;

        qselect(mth_array, mth_indeces, n, k-1);
        memcpy(ret.ndist + i*k, mth_array, sizeof(double) * k);
        memcpy(ret.nidx + i*k, mth_indeces, sizeof(uint32_t) * k);

        free(mth_indeces);
#endif

#if TRANS == 0 && MATRIX == 1
        free(mth_array);
#endif
    }

#if TRANS == 1
    free(distance_trans);
#elif TRANS == 0
    free(distance);
#endif

    TOC("Time elapsed calculating kNN given distance matrix (seconds): %lf\n");

    return ret;
}


// D = sqrt(sum(X.^2,2) - 2 * X*Y.' + sum(Y.^2,2).');
double *euclidean_distance(double *x, double *y, uint32_t n, uint32_t d, uint32_t m) {
    printf("Calculating distance matrix nxm approach\n");
    struct timespec tic;
    struct timespec toc;

    TIC();

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

    TOC("Time elapsed calculating dot product (seconds): %lf\n");

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

// D' = sqrt(sum(X.^2,2).' - 2 * Y * X.' + sum(Y.^2, 2))
double *euclidean_distance_notrans(double *x, double *y, uint32_t n, uint32_t d, uint32_t m) {
    printf("Calculating distance matrix mxn approach\n");
    struct timespec tic;
    struct timespec toc;

    TIC();

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

    TOC("Time elapsed calculating dot product (seconds): %lf\n");

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
#define SWAPval(a, b) { tmp1 = v[a]; v[a] = v[b]; v[b] = tmp1; }
#define SWAPidx(a, b) { tmp2 = idx[a]; idx[a] = idx[b]; idx[b] = tmp2; }

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
