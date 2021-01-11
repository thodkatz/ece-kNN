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

/* 
 * \brief Divide an array nxd to a fair amount to each process and track the size and the memory displacement
 *
 */
void memdistr(int n, int d, int numtasks, int *size_per_proc, int *memory_offset);

/*
 * \brief Shift elements of an array one time to the left
 */
void rotate_left(int *arr, int size);

/*
 * \brief For a given matrix divided to parts, for each part adjust the indeces as viewed by the total matrix
 */
void adjust_indeces(uint32_t *arr, uint32_t row, uint32_t col, int offset);

#endif
