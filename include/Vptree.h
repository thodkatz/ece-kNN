#ifndef VPTREE_H
#define VPTREE_H

#include <stdint.h>
#include <queue>

struct Node {
    uint32_t index; 
    double mu;
    Node *left;
    Node *right;
    Node() : index(0), mu(0.), left(NULL), right(NULL) {}
};

class Vptree {

    private:
        double   *_corpus;
        uint32_t *_indeces;
        uint32_t _dimensions;
        double   _tau;
        double   *_target;
        uint32_t _k;

        std::priority_queue<std::pair<double, uint32_t>> heap;

        Node *_root;

        Node *makeTree(uint32_t low, uint32_t high);

        void searchTree(Node *node);

        void select_vp(double *cords, uint32_t *index, double *corpus, int low, int high);

        double *euclidean_dist_naive_point_with_corpus(double *y, int low, int high);

        // a modified quciselect. Based on the distance, the index and the corpus are rearranged
        double kselect_dist_corpus_index(double *dist, double *corpus, uint32_t *indeces, uint32_t low, int64_t len, int64_t k);
    
    public:
        // create a vp tree 
        Vptree(double *corpus, uint32_t *indeces, uint32_t n, uint32_t dimensions);

        //search
        void searchKNN(double *dist, uint32_t *idx, double *target, uint32_t k);
};

#endif
