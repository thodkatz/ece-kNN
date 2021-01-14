#ifndef V1_H
#define V1_H

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

#endif
