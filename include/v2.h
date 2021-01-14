#ifndef V2_H
#define V2_H

#include "Vptree.h"
#include <stdint.h>

/*
 * Define process with rank zero (root) as MASTER
 */
#define MASTER 0

//! Compute distributed all-kNN of points in X
/*!

  \param  X      Data points                     [n-by-d]
  \param  n      Number of data points           [scalar]
  \param  d      Number of dimensions            [scalar]
  \param  k      Number of neighbors             [scalar]

  \return  The kNN result
*/
knnresult distrAllkNN(double *x, uint32_t n, uint32_t d, uint32_t k, int argc, char *argv[]);

/*
 * Search vp-tree \p vpt for query (target) points \p y
 */
knnresult kNN_vptree(Vptree &vpt, double *y, int n, uint32_t m, uint32_t d, uint32_t k);

#endif 
