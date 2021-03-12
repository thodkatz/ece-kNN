<h1> Table of contents </h1>

- [Description](#description)
  - [V0 Sequential](#v0-sequential)
  - [V1 Asynchronous](#v1-asynchronous)
  - [V2 Avoid computing all-to-all distances](#v2-avoid-computing-all-to-all-distances)
  - [What to submit](#what-to-submit)
  - [Afternotes](#afternotes)
- [Report](#report)
- [Compile and run](#compile-and-run)
  - [Options](#options)
- [Output](#output)
- [Validation](#validation)



# Description

Implement in MPI a distributed all-KNN search and find algorithm for the k nearest neighbors (k-NN) of each point x∈X.

The set of X points will be passed to you as an input array along with the number of points n, the number of dimensions d and the number of neighbors k.

Each MPI process Pi will calculate the distance of its own points from all other points and record the distances and indices of the k nearest for each of its own points.

## V0 Sequential

Write the sequential version, which finds for each point in a query set Y the k nearest neighbors in the corpus set X, according to this spec

```
// Definition of the kNN result struct
typedef struct knnresult{
  int    * nidx;    //!< Indices (0-based) of nearest neighbors [m-by-k]
  double * ndist;   //!< Distance of nearest neighbors          [m-by-k]
  int      m;       //!< Number of query points                 [scalar]
  int      k;       //!< Number of nearest neighbors            [scalar]
} knnresult;

//! Compute k nearest neighbors of each point in X [n-by-d]
/*!

  \param  X      Corpus data points              [n-by-d]
  \param  Y      Query data points               [m-by-d]
  \param  n      Number of corpus points         [scalar]
  \param  m      Number of query points          [scalar]
  \param  d      Number of dimensions            [scalar]
  \param  k      Number of neighbors             [scalar]

  \return  The kNN result
*/
knnresult kNN(double * X, double * Y, int n, int m, int d, int k);

```

Note: All two-dimensional matrices are stored in row-major format.

To calculate an m×n Euclidean distance matrix D between two sets points X and Y of m and n points respectively, use the following operations

like in this MATLAB line:
`D = sqrt(sum(X.^2,2) - 2 * X*Y.' + sum(Y^2,2).');`

Note: In older MATLAB versions, the above command with raise an error of dimension mismatch, since singleton expansion was not supported. Use bsxfun instead.

Hint: Use high-performance BLAS routines, for matrix multiplication. For large values of m, n, block the query Y and compute the distance matrix one block at a time, to avoid storing large intermediate distance matrices.

After that, only keep the k shortest distances and the indices of the corresponding points (using k-select).

## V1 Asynchronous

Use your V0 code to run as p MPI processes and find the all kNN of the points that are distributed in disjoint blocks to all processes, according to the following spec

```
//! Compute distributed all-kNN of points in X
/*!

  \param  X      Data points                     [n-by-d]
  \param  n      Number of data points           [scalar]
  \param  d      Number of dimensions            [scalar]
  \param  k      Number of neighbors             [scalar]

  \return  The kNN result
*/
knnresult distrAllkNN(double * X int n, int d, int k);

```

We move the data along a ring, (receive from previous and send to the next process) and update the k nearest every time.

- How many times do we need to iterate to have all points checked against all other points?
- Hide all communication costs by transferring data using asynchronous communications, while we compute.
- How the total run time of V1 compares with the total time of V0?
- Assume that you can fit in memory the local points X (corpus), the points Y (query) you are working with, and space for the incoming (query) points Z.
- Use pointers to exchange the locations Y and Z

## V2 Avoid computing all-to-all distances

Build a Vantage Point Tree (VPT) with the local corpus data points. Stop when leaves contain up to B>1 points, for a reasonable choice value. Extend V1 to V2 where you pre-compute the VPT of the points X locally and use them to query with the working sets Y. In this version, you will pass the VPT along the ring.

- Compare run times and make sure you agree with the V1 results

## What to submit

- A 3-page report in PDF format (any pages after the 3rd one will not be taken into account). Report execution times of your implementations with respect to the number of data points n, the number of dimensions d, and the number of processes p. Use 10,000sqrt(p)≤n≤25,000sqrt(p) and 3≤d≤20 and 10≤k≤100. Test with both random matrices, and real-data from https://archive.ics.uci.edu/ml/datasets.php.
- Upload the source code on GitHub, BitBucket, Dropbox, Google Drive, etc. and add a link in your report.
- Check the validation of your code using the automated tester on e-learning.

## Afternotes
We can build a global VPT without broadcasting the local points but only the vantage point and median distance at each tree node. Then we can store and use the in-out sequence “signature” of each point to zoom in directly to the subtrees that can provably contain the neighbors and exclude other subtrees.

The VPT construction can be parallelized locally within each process, using Pthreads, Cilk, or openMP.

# Report

[Parallel k-Nearest Neighbors](report/report.pdf)

# Compile and run

``` shell
$ make <version>
```  
Available targets: v0 v1 v2  
e.g.   
``` shell
$ make v0
```  
  
Version 0 doesn't use MPI routines (serial)  
``` shell
$ ./bin/v0
```  
For v1 and v2:  
``` shell
$ mpirun -n <number of processors> ./bin/<version> <dataset> <number of neighbors>
```  
e.g.   
```
$ mpirun -n 4 ./bin/v1 datasets/corel/CorelMoments.asc 10
```  

## Options

- For large input data configure the number of blocks for v0 editing the macro `#define BLOCKS <number of blocks>` in `src/v0/main.c`.
- For testing with random matrices for v0, v1, v2 edit (comment/uncomment) the macro `#define RANDOM` in `src/v0/main.c`, `src/v1/v1.c` and `src/v2/v2.c` respectively.
- For testing different communication models regarding MPI for v1 edit (comment/uncomment) the macro `#define RING` in `src/v1/v1.c`
- Regarding vp-tree, control the height of the tree by `const float target_height_tree_percent = <desired value 0-1>` in `src/v2/v2.c`. By default is 1.0 (full balanced tree)
- Vantage point can be selected with three ways. Navigate in `src/v2/Vptree.cpp` and in `makeTree()` edit the corresponding lines to test different selection methods.

# Output

Each time you run a version, you can see its output under `logs/`. Each log file contains the distances appended with the indices of the k nearest neighbors.

# Validation

After running the program for all the three versions, run the Matlab script. 