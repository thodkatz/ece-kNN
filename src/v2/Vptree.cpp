#include <stdint.h>
#include "Vptree.h"
#include "main.h"
#include <algorithm>
#include <limits>
#include <queue>
#include <string.h>
#include <math.h>
#include <stdlib.h>

Vptree::Vptree(double *corpus, uint32_t *indeces, uint32_t n, uint32_t dimensions) {
    _dimensions = dimensions;
    _corpus = corpus;
    _indeces = indeces;
    
    /* printf("The dim are: %d\n", _dimensions); */
    /* print_dataset_yav(_corpus, n, _dimensions); */
    /* printf("\n"); */
    /* for(uint32_t i = 0; i < n; i++) { */
    /*     _indeces[i] = i; */
    /*     printf("%d ", _indeces[i]); */
    /* } */
    /* printf("\n"); */

    _root = Vptree::makeTree(0, n);

    printf("Final indeces and corpus\n");
    print_dataset_yav(_corpus, n, _dimensions);
    printf("\n");
    for(uint32_t i = 0; i < n; i++) {
        printf("%d ", _indeces[i]);
    }
    printf("\n");
}

Node *Vptree::makeTree(uint32_t low, uint32_t high) {
#define SWAPcorpus(a, b) { \
    memcpy(tmp3, _corpus + a, sizeof(double)*_dimensions); memcpy(_corpus + a, _corpus + b, sizeof(double)*_dimensions); memcpy(_corpus + b, tmp3, sizeof(double)*_dimensions); }

    printf("\nLets make a tree (Christmas?)\n");
    printf("The low is %d and the high is %d\n", low, high);

    int points_corpus = high - low;
    printf("The points are: %d\n", points_corpus);

    if(points_corpus == 0) return NULL;

    Node *vp = new Node();
    
    printf("The indices are:\n");
    for(uint32_t i = low; i < high; i++) {
        printf("%d ", _indeces[i]);
    }
    printf("\n");

    printf("The dataset is\n");
    if(points_corpus < 0) {printf("Negative pointsss\n"); exit(1);}
    print_dataset_yav_range(_corpus, low, high, _dimensions);

    if(points_corpus == 1) {
        printf("We have a leaf!!\n");
        vp->index = low;
        printf("The vantage point index is %d and corresponds to the %d element of the global corpus\n", vp->index, _indeces[vp->index]);
        vp->mu = 0;

        return vp;
    }

    //select_vp(vp_cords, vp.index, low, high);
    
    srand(1);
    int temp = rand()%points_corpus + low;
    vp->index = temp;
    printf("\nThe random number is %d and index: %d\n", temp, _indeces[temp]);
    double *vp_cords;
    MALLOC(double, vp_cords, _dimensions);
    memcpy(vp_cords, _corpus + temp * _dimensions, sizeof(double) * _dimensions); 
    
    print_dataset_yav(vp_cords, 1, _dimensions);
    printf("\n");

    // the dist now will have the first element the vantage point
    double *tmp3;
    MALLOC(double, tmp3, sizeof(double)*_dimensions);
    SWAPcorpus(temp * _dimensions, low * _dimensions);
    free(tmp3);

    printf("Swapping distance\n");
    if(points_corpus < 0) {printf("bad whyyy?\n"); exit(1);}
    print_dataset_yav_range(_corpus, low, high, _dimensions);
    printf("\n");

    printf("Swapping indeces\n");
    std::swap(_indeces[temp], _indeces[low]);
    for(uint32_t i = low; i < high; i++) {
        printf("%d ", _indeces[i]);
    }
    printf("\n");

    vp->index = low;
    printf("The vantage point index is %d and corresponds to the %d element of the global corpus\n", vp->index, _indeces[vp->index]);

    double *dist;
    dist = Vptree::euclidean_dist_naive_point_with_corpus(vp_cords, low, high);
    print_dataset_yav(dist, 1, points_corpus);
    printf("\n");
    free(vp_cords);

    points_corpus--;
    uint32_t median = (points_corpus)/2; // if even, unbalanced tree
    printf("The median is %d\n", median);

    // exclude vantage point itself
    if(points_corpus!=1) vp->mu = Vptree::kselect_dist_corpus_index(dist + 1, _corpus, _indeces, low + 1, points_corpus, median);
    else vp->mu = dist[1];

    printf("Rearanged dist\n");
    print_dataset_yav(dist, 1, points_corpus+1);
    printf("The radius/threshold is %f\n", vp->mu);
    printf("Index of the radius threshold %d\n", _indeces[low+1+median]);
    printf("Partition corpus based on distance\n");
    print_dataset_yav_range(_corpus, low, high, _dimensions);
    printf("\n");
    for(uint32_t i = low; i < high; i++) {
        printf("%d ", _indeces[i]);
    }
    printf("\n");

    //vp->index = _indeces[median + low + 1];

    free(dist);

    vp->left = makeTree(low + 1, low + 1 + median);

    vp->right = makeTree(low + 1 + median, high);

    return vp;
}

void Vptree::searchTree(Node *node) {

    if(node == NULL) return;

    double *dist = Vptree::euclidean_dist_naive_point_with_corpus(_target, node->index, node->index+1);

    printf("The index %d and the corpus %d and the dist %f\n", node->index, _indeces[node->index], *dist);

    std::pair<double, uint32_t> dist_index;

    if (*dist < _tau) {
        if (heap.size() == _k) heap.pop();
        dist_index.first = *dist;
        dist_index.second = _indeces[node->index];

        heap.push(dist_index);

        if (heap.size() == _k) _tau = heap.top().first;
    }

    // Leaf
    if (node->left == NULL && node->right == NULL) {
        return;
    }

    if(*dist + _tau >=  node->mu) searchTree(node->right);
    if(*dist - _tau <= node->mu) searchTree(node->left);
}

void Vptree::searchKNN(double *dist, uint32_t *idx, double *target, uint32_t k) {
    _tau = std::numeric_limits<double>::max();
    _k = k;
    _target = target; // 1xd

    printf("Traget\n");
    print_dataset_yav(_target, 1 ,_dimensions);

    searchTree(_root);

    int isFirst = 1;
    uint32_t count = k-1;
    while (!heap.empty()) {
        dist[count] = heap.top().first;
        idx[count] = heap.top().second + isFirst;

        count--;
        heap.pop();
    }

}

void Vptree::select_vp(double *cords, uint32_t *index, double *x, int n, int d) {

}

// naive euclidean distance matrix.
double *Vptree::euclidean_dist_naive_point_with_corpus(double *query, int low, int high) {
    //printf("Calculating distance matrix naive approach\n");

    int points = high - low;
    double *distance;
    MALLOC(double, distance, points);
    uint32_t count = 0;

    for(uint32_t j = low; j < high; j++) {
        double temp = 0;
        for (uint32_t k = 0; k < _dimensions; k++) {
            temp += (_corpus[j*_dimensions + k] - query[k]) * (_corpus[j*_dimensions + k] - query[k]);
        }
        distance[count] = sqrt(temp);
        count++;
    }

    return distance;
}

double Vptree::kselect_dist_corpus_index(double *dist, double *corpus, uint32_t *indeces, uint32_t low, int64_t len, int64_t k) {
#define SWAPdist(a, b) { tmp1 = dist[a]; dist[a] = dist[b]; dist[b] = tmp1; }
#define SWAPindeces(a, b) { tmp2 = _indeces[a]; _indeces[a] = _indeces[b]; _indeces[b] = tmp2; }
#define SWAPcorpus(a, b) { \
    memcpy(tmp3, _corpus + a, sizeof(double)*_dimensions); memcpy(_corpus + a, _corpus + b, sizeof(double)*_dimensions); memcpy(_corpus + b, tmp3, sizeof(double)*_dimensions); }

	int64_t i, st;
    double tmp1;
    uint32_t tmp2;
    double *tmp3;
    MALLOC(double, tmp3, sizeof(double) * _dimensions);
 
	for (st = i = 0; i < len - 1; i++) {
		if (dist[i] > dist[len-1]) continue;
		SWAPdist(i, st);
		SWAPindeces(i + low, st + low);
		SWAPcorpus((i + low)*_dimensions, (st + low)*_dimensions);
		st++;
	}
 
	SWAPdist(len-1, st);
	SWAPindeces(len-1 + low, st + low);
	SWAPcorpus((len-1 + low)*_dimensions, (st + low)*_dimensions);
 
    free(tmp3);

	return k == st	?dist[st] 
			:st > k	? kselect_dist_corpus_index(dist, _corpus, _indeces, low, st, k)
				: kselect_dist_corpus_index(dist + st, _corpus + st * _dimensions, _indeces + st, low + st, len - st, k - st);
}
