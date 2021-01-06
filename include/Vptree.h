#ifndef VPTREE_H
#define VPTREE_H

#include <stdint.h>
#include <queue>

/*
 * Required input for sending a vantage point tree via MPI
 *
 */ 
typedef struct MPI_Tree {
    double *vp_mu;
    double *vp_coords;
    int *vp_index;
}MPI_Tree;

class Vptree {

    public:

        // Node of the Vptree
        double     *vp_mu;                 // Radius/median for every node
        double     *vp_coords;             // Coordinates of vantage point
        int        *vp_index;              // Index of vantage point that corresponds to the index of a point of the local corpus
        int total_nodes_visited  = 0; // Total numbers visited to find kNN. This will can be compared with the _n x _n

    private:
        int _height_tree;        // The height of the tree
        int _target_height_tree; // Don't create the whole tree. Stop until _targetHeightTree
        int _num_nodes_balanced; // The number of nodes of the tree created by the corpus points if it was balanced 
        int _count_nodes;         // count how many nodes we visited. Should be less or equal of the total 

        uint32_t _dimensions;  // The number of dimensions of the metric space
        int      _n;
        double   *_corpus;     // Points of the corpus. Each point has _dimensions values
        uint32_t *_indeces;    // The indeces that correspond to each corpus point
        double   _tau;         // furthest distance from vp
        double   *_target;     // The coordinates of each point that we evaluate as a neighbor
        uint32_t _k;           // The number of neighbors

        std::priority_queue<std::pair<double, int>> heap; // The distances of each query points ordered

        /*
         * Create a tree recursively in depth first format [root, left, right, ...]
         *
         * Each node has its corresponding data stored in the public members corpus, indeces and mu
         *
         * \param low
         * \param high
         * \param index_node The index of each node in the serialized version of the tree
         */
        void makeTree(int height, uint32_t low, uint32_t high, int index_node);

        /*
         * Search a tree in depth first format
         *
         * \param height
         * \param index_node
         */
        void searchTree(int height, int index_node);

        /*
         * Evaluate the best vantage point.
         *
         * For the root we picked the vantage point based on the variance given two random samples. For the rest we picked the furthest vantage point from the previous
         *
         * \param
         * \param
         * \param
         * \return The row index (coordinates) of the corpus
         */
        //void select_vp(double *cords, uint32_t *index, double *corpus, int low, int high);
        int select_vp(int low, int high, int index_node);

        /*
         * For the target node, calcuate all the distances to the given local corpus
         *
         */
        double *point_with_corpus(double *y, double *corpus, int low, int high);

        /*
         * Euclidean distance between two points
         *
         * \param x The first point (1xd array)
         * \param y The second point (1xd array)
         */
        double points_distance(double *x, double *y);

        /*
         * A modified quciselect. Based on the distance, the index and the corpus are rearranged
         *
         * TODO: Compate with std::nth_element C++ utility
         */
        double kselect_dist_corpus_index(double *dist, double *corpus, uint32_t *indeces, uint32_t low, int64_t len, int64_t k);

        /*
         * Swap rows (1d array) of a 2d array given in row format
         *
         * \param index_first The index of the first element of the 2d array. The column size should be considered.
         * \param index_second The index of the second element of the 2d array. The column size should be considered.
         */
        void swap_row(int index_first, int index_second, double *array, int cols);

        void sample_and_indeces(double *vals, uint32_t *indeces, int num, int low, int high);

        void sample(double *vals, int num, int low, int high);

        int variance_select_vp(int low, int high);

    public:

        /* 
         * Each process before searching the tree if it isn't created explicitly (constructor calling the makeTree()), should initialize the required data structures
         *
         * Search the tree skiping the creation, provided the required data structure as arguments.
         *
         * \param n
         * \param height_tree
         * \param target_height_percent
         */
        void init_before_search(int n, int num_nodes_balanced, int height_tree, float target_height_percent, double *vp_mu, double *vp_coords, int *vp_index);

        /* 
         * At the beginning, each process should create its own tree based on the input scattered by the master process 
         *
         * This constructor should be called once, in the tree creation. 
         *
         * \param corpus
         * \param indeces
         * \param n
         * \param d Dimensions of the metric space
         * \param target_height_percent The percentage of the height tree
         */
        Vptree(double *corpus, uint32_t *indeces, uint32_t n, uint32_t dimensions, int num_nodes_balanced, int height_tree, float target_height_percent);

        /*
         * Search for k-Nearest Neighbors
         */
        void searchKNN(double *dist, uint32_t *idx, double *target, uint32_t k);
};

#endif
