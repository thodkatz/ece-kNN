#ifndef MAIN_H
#define MAIN_H

#include <stdint.h>
#include <stdio.h>

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

#endif
