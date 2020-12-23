#ifndef MAIN_H
#define MAIN_H

#include <stdint.h>

#define MALLOC(type, x, len) if((x = (type*)malloc(len * sizeof(type))) == NULL) \
                                {printf("Bad alloc\n"); exit(1);}

/*
 * Required to create two objects of struct timespec tic, toc
 */
#define TIC() clock_gettime(CLOCK_MONOTONIC, &tic);
#define TOC(text) clock_gettime(CLOCK_MONOTONIC, &toc); \
                printf(text, diff_time(tic,toc));

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

// color because why not
#define CYN   "\x1B[36m"
#define RED   "\x1B[31m"
#define RESET "\x1B[0m"

// Definition of the kNN result struct
typedef struct knnresult{
  uint32_t    * nidx;    //!< Indices (0-based) of nearest neighbors [m-by-k]
  double      * ndist;   //!< Distance of nearest neighbors          [m-by-k]
  uint32_t      m;       //!< Number of query points                 [scalar]
  uint32_t      k;       //!< Number of nearest neighbors            [scalar]
} knnresult;


//! Compute k nearest neighbors of each point in X [n-by-d]
/*!

  Note: If the query point is one of the corpus set points then it is included as the first neighbor.

  \param  x      Corpus data points              [n-by-d]
  \param  y      Query data points               [m-by-d]
  \param  n      Number of corpus points         [scalar]
  \param  m      Number of query points          [scalar]
  \param  d      Number of dimensions            [scalar]
  \param  k      Number of neighbors             [scalar]

  \return  The kNN result
  */
knnresult kNN(double *x, double *y, uint32_t n, uint32_t m, uint32_t d, uint32_t k);

//! Compute distributed all-kNN of points in X
/*!

  \param  X      Data points                     [n-by-d]
  \param  n      Number of data points           [scalar]
  \param  d      Number of dimensions            [scalar]
  \param  k      Number of neighbors             [scalar]

  \return  The kNN result
*/
knnresult distrAllkNN(double *x, uint32_t n, uint32_t d, uint32_t k);

/*
 * \brief Transpose matrix
 */
void transpose(double *src, double *dst, const uint32_t N, const uint32_t M);



/*
 * \brief Naive quickselect implementation
 *
 * source: geeksforgeeks
 *
 */
int64_t quickselect(double arr[], int64_t start, int64_t end, int64_t k);

/*
 * \brief Parition array utilized for the quickselect
 *
 * Note: Use median of 3 to optimize performance
 */
uint32_t partition(double arr[], uint32_t start, uint32_t end);

/*
 * \brief Naive quickselect implementation
 *
 * Note: Returns the indeces of the k smalled values of a array \p v
 *
 * source: rosettacode database
 */
double qselect(double *v, uint32_t *idx, int64_t len, int64_t k);

double *euclidean_distance(double *x, double *y, uint32_t n, uint32_t d, uint32_t m);

double *euclidean_distance_naive(double *x, double *y, uint32_t n, uint32_t d, uint32_t m);

double *euclidean_distance_notrans(double *x, double *y, uint32_t n, uint32_t d, uint32_t m);

/*
 * \brief Elapsed time between two reference points using monotonic clock
 *
 * \return Elapsed time in seconds
 */
double diff_time(struct timespec, struct timespec);

/*
 * \brief Matrix Market format to COO
 *
 * source: https://math.nist.gov/MatrixMarket/mmio/c/example_read.c
 *
 */
void mm2coo(int argc, char* argv[], uint32_t **rows, uint32_t **columns, uint32_t nnz, uint32_t n);

/*
 * \brief Print 1d array using ; sepatated values format
 *
 * Format:
 * [a ; b ; c ; d]
 *
 */
void print_dataset(double *array, uint32_t row, uint32_t col);
// void print_dataset_semicolon

/*
 * \brief Print 2d array using comma separated values format
 *
 * Note: Type of values double
 *
 * Format:
 * [[a, b, b]
 *  [d, e, f]
 *  [g, h, j]]
 */
void print_dataset_yav(double *array, uint32_t row, uint32_t col);
// void print_dataset_comma

/*
 * \brief Print 2d array using comma separated values format
 *
 * Note: Type of values uint32_t
 *
 * Format:
 * [[a, b, b]
 *  [d, e, f]
 *  [g, h, j]]
 */
void print_indeces(uint32_t *array, uint32_t row, uint32_t col);

#endif
