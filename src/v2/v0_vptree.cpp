#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "main.h"
#include "v2.h"
#include "Vptree.h"


/*
 * Calculate dot product using cblas routine
 *
 * 0 --> false
 * 1 --> true 
 */
#define CBLAS_DDOT 0

// assume that kNN including itself 
knnresult kNN_vptree(Vptree &vpt, double *y, int n, uint32_t m, uint32_t d, uint32_t k) {

    printf("Entering knn function, m:%d\n", m);
    printf("The y is \n");
    print_dataset_yav(y, m, d);

    knnresult ret;
    MALLOC(double, ret.ndist, m*MIN(k, n));
    MALLOC(uint32_t, ret.nidx, m*MIN(k, n));
    ret.k = MIN(n,k); 
    ret.m = m;

    struct timespec tic;
    struct timespec toc;
    
    TIC()

    // search vp tree
    for(uint32_t i = 0; i < m; i++) {
        double *target;
        MALLOC(double, target, d);
        memcpy(target, y + i*d, sizeof(double) * d);
        vpt.searchKNN(ret.ndist + MIN(k,n)*i, ret.nidx + MIN(k,n)*i, target, MIN(n,k));
        free(target);

        printf("\nDistance of kNN\n");
        print_dataset_yav(ret.ndist + m*i, 1, MIN(n,k));
        printf("\nIndeces of kNN\n");
        print_indeces(ret.nidx + m*i, 1, MIN(n,k));
    }

    TOC("Time elapsed calculating kNN vp-tree (seconds): %lf\n")
    return ret;
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
