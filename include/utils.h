#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include <stdio.h>

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

/*
 * \brief Elapsed time between two reference points using monotonic clock
 *
 * \return Elapsed time in seconds
 */
double diff_time(struct timespec, struct timespec);

double *read_matrix(int *n, int *d, int argc, char *argv[]);

double *read_corel(FILE *f, char *file_name, int *n, int *d);

double *read_features(FILE *f, int *n, int *d);

double *read_mini(FILE *f, int *n, int *d);

double *read_tv(FILE *f, char *file_name, int *n, int *d);

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
double qselect_and_indeces(double *v, uint32_t *idx, int64_t len, int64_t k);

double qselect(double *v, int64_t len, int64_t k);

/*
 * \brief Get the three indeces of an array and return index of the median
 *
 * \param array The array that its values is compared given the indeces
 */
int medianThree(double *array, int a, int b, int c);

double *euclidean_distance(double *x, double *y, uint32_t n, uint32_t d, uint32_t m);

double *euclidean_distance_naive(double *x, double *y, uint32_t n, uint32_t d, uint32_t m);

double *euclidean_distance_notrans(double *x, double *y, uint32_t n, uint32_t d, uint32_t m);

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

/*
 * Print to file the indeces and the distance matrix
 *
 */
void print_output_file(FILE *f, double *dist, uint32_t *indeces, uint32_t row, uint32_t col);

/*
 * Print to file the input array
 *
 */
void print_input_file(FILE *f, double *array, uint32_t row, uint32_t col);

#endif
