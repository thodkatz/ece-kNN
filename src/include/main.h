#ifndef MAIN_H
#define MAIN_H

#include <stdint.h>

// Definition of the kNN result struct
typedef struct knnresult{
  uint32_t    * nidx;    //!< Indices (0-based) of nearest neighbors [m-by-k]
  double      * ndist;   //!< Distance of nearest neighbors          [m-by-k]
  uint32_t      m;       //!< Number of query points                 [scalar]
  uint32_t      k;       //!< Number of nearest neighbors            [scalar]
} knnresult;


//! Compute k nearest neighbors of each point in X [n-by-d]
/*!

  Note: Unexpected behaviour if there are multiple instances of the kth elements

  \param  x      Corpus data points              [n-by-d]
  \param  y      Query data points               [m-by-d]
  \param  n      Number of corpus points         [scalar]
  \param  m      Number of query points          [scalar]
  \param  d      Number of dimensions            [scalar]
  \param  k      Number of neighbors             [scalar]

  \return  The kNN result
  */
knnresult kNN(double *x, double *y, uint32_t n, uint32_t m, uint32_t d, uint32_t k);

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

#endif
