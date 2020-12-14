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

/**
 *  \brief COO to CSC conversion
 *
 *  Converts a square matrix from COO to CSC format.
 *
 *  Note: The routine assumes the input COO and the output CSC matrix
 *  to be square.
 *
 */
void coo2csc(
  uint32_t       * const row,       /*!< CSC row start indices */
  uint32_t       * const col,       /*!< CSC column indices */
  uint32_t const * const row_coo,   /*!< COO row indices */
  uint32_t const * const col_coo,   /*!< COO column indices */
  uint32_t const         nnz,       /*!< Number of nonzero elements */
  uint32_t const         n,         /*!< Number of rows/columns */
  uint32_t const         isOneBased /*!< Whether COO is 0- or 1-based */
); 

#endif
