#ifndef V1_H
#define V1_H

/*
 * Define process with rank zero (root) as MASTER
 */
#define MASTER 0

/*
 * \brief Divide an array nxd to a fair amount to each process and track the size and the memory displacement
 *
 */
void memdistr(uint32_t n, uint32_t d, int numtasks, int *size_per_proc, int *memory_offset);

/*
 * \brief Shift elements of an array one time to the left
 */
void rotate_left(int *arr, int size);

/*
 * \brief For a given matrix divided to parts, for each part adjust the indeces as viewed by the total matrix
 */
void adjust_indeces(uint32_t *arr, uint32_t row, uint32_t col, int offset);

#endif
