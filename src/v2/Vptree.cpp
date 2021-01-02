#include <stdint.h>
#include "Vptree.h"
#include "main.h"
#include <algorithm>
#include <limits>
#include <queue>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <tuple>
#include <iostream>
#include <array>

#define FLAG_NO_LEAF -1
#define LEFT_CHILD(index) index*2 + 1
#define RIGHT_CHILD(index) index*2 + 2

void Vptree::init_before_search(int n, int num_nodes_balanced, int height_tree, float target_height_percent, double *vp_mu, double *vp_coords, int *vp_index) {
    _num_nodes_balanced = num_nodes_balanced;
    _height_tree = height_tree;
    _n = n;

    this->vp_mu = vp_mu;
    this->vp_coords = vp_coords;
    this->vp_index = vp_index;

    printf("Entering init before search\n");
    for(int i = 0; i < _num_nodes_balanced; i++) {
        printf("%d node\n", i);
        printf("The radius %f\n", vp_mu[i]);
        printf("The index %d \n", vp_index[i]);
        printf("The coords\n");
        for (int j = 0; j < _dimensions; j++) printf("%f ", vp_coords[i*_dimensions + j]);
        printf("\n");
    }

    /* assert(n>1); */
    /* height_tree = ceil(log(n-1)); // assume root = 0th height */
    printf("The height of the complete tree is %d\n", height_tree);

    _target_height_tree = _height_tree * target_height_percent;
    assert(0 < _target_height_tree && _target_height_tree <= _height_tree);
    printf("The height percentage is %f\n", target_height_percent);
    printf("The requested (rounded) height of the tree is %d\n", _target_height_tree);
}

Vptree::Vptree(double *corpus, uint32_t *indeces, uint32_t n, uint32_t dimensions, int num_nodes_balanced, int height_tree, float target_height_tree_percent) {
    _dimensions = dimensions;
    _corpus = corpus;
    _indeces = indeces;
    _n = n;
    _num_nodes_balanced = num_nodes_balanced;
    _height_tree = height_tree;

    
    /* printf("The dim are: %d\n", _dimensions); */
    /* print_dataset_yav(_corpus, n, _dimensions); */
    /* printf("\n"); */
    /* for(uint32_t i = 0; i < n; i++) { */
    /*     _indeces[i] = i; */
    /*     printf("%d ", _indeces[i]); */
    /* } */
    /* printf("\n"); */

    //_root = Vptree::makeTree(0, n);

    // ceil n to a power of 2
    // For the tree created by n, what is its full balanced?
    // -1 will be our flag to indicate if there aren't any leaves
    //assert(n>1);
    //height_tree = ceil(log(n-1)); // assume root = 0th height
    _target_height_tree = _height_tree * target_height_tree_percent;
    assert(0 < _target_height_tree && _target_height_tree <= _height_tree);
    printf("The height of the complete tree is %d\n", _height_tree);
    printf("The height percentage is %f\n", target_height_tree_percent);
    printf("The requested (rounded) height of the tree is %d\n", _target_height_tree);

    /* _num_nodes_balanced = pow(2, _height_tree) + 1; */
    /* printf("Points needed to complete a balanced tree %d\n", _num_nodes_balanced - n); */

    // init each node of the vptree
    MALLOC(double, vp_mu, _num_nodes_balanced);
    MALLOC(double, vp_coords, _num_nodes_balanced * dimensions);
    MALLOC(int, vp_index, _num_nodes_balanced);
    for(int i = 0; i < _num_nodes_balanced; i++) {
        vp_mu[i] = FLAG_NO_LEAF;
        vp_index[i] = FLAG_NO_LEAF;
        for(int j = 0; j < dimensions; j++) vp_coords[i*_num_nodes_balanced + j] = 0;
    }

    Vptree::makeTree(0, n, 0);

    printf("Final indeces and corpus\n");
    print_dataset_yav(_corpus, n, _dimensions);
    printf("\n");
    for(uint32_t i = 0; i < n; i++) {
        printf("%d ", _indeces[i]);
    }
    printf("\n");

    printf("Final indeces and corpus\n");
    print_dataset_yav(vp_coords, _num_nodes_balanced, _dimensions);
    printf("\n");
    for(int i = 0; i < _num_nodes_balanced; i++) {
        printf("%d ", vp_index[i]);
    }
    printf("\n");
}

void Vptree::makeTree(uint32_t low, uint32_t high, int index_node) {
//#define SWAPcoords(a, b) { \
    memcpy(tmp, _corpus + a, sizeof(double)*_dimensions); memcpy(_corpus + a, _corpus + b, sizeof(double)*_dimensions); memcpy(_corpus + b, tmp, sizeof(double)*_dimensions); }
#define SWAPindeces(a, b) { tmp = _indeces[a]; _indeces[a] = _indeces[b]; _indeces[b] = tmp; }

    printf("\nLets make a tree (Christmas?)\n");
    printf("The low is %d and the high is %d\n", low, high);

    int points_corpus = high - low;
    printf("The points are: %d\n", points_corpus);

    if(points_corpus == 0) return; 
    
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
        memcpy(vp_coords + index_node*_dimensions, _corpus + low*_dimensions, sizeof(double)*_dimensions);
        print_dataset_yav(vp_coords + index_node*_dimensions, 1, _dimensions);
        //vp->index = low;
        //NodeData.second = low;
        vp_index[index_node] = _indeces[low];

        printf("The vantage point corresponds to the %d element of the global corpus\n", vp_index[index_node]);
        //vp->mu = 0;
        //NodeData.first = 0;

        return;
    }

    //select_vp(vp_coords, vp.index, low, high);
    srand(1);
    int temp = rand()%points_corpus + low;
    //vp->index = temp;
    //NodeData.second = temp;
    printf("\nThe random number is %d and index: %d\n", temp, _indeces[temp]);
    
    // the dist now will have the first element the vantage point
    Vptree::swap_row(temp*_dimensions, low*_dimensions, _corpus, _dimensions);

    printf("Swapping distance\n");
    if(points_corpus < 0) {printf("bad whyyy?\n"); exit(1);}
    print_dataset_yav_range(_corpus, low, high, _dimensions);
    printf("\n");
    memcpy(vp_coords + index_node*_dimensions, _corpus + low*_dimensions, sizeof(double) * _dimensions);
    print_dataset_yav(vp_coords + index_node*_dimensions, 1, _dimensions);
    printf("\n");

    printf("Swapping indeces\n");
    //std::swap(_indeces[temp], _indeces[low]);
    uint32_t tmp;
    SWAPindeces(temp, low);
    for(uint32_t i = low; i < high; i++) {
        printf("%d ", _indeces[i]);
    }
    printf("\n");
    vp_index[index_node] = _indeces[low];


    //vp->index = low;
    //NodeData.second = low;
    printf("The vantage point corresponds to the %d element of the global corpus\n", vp_index[index_node]);

    double *dist;
    dist = Vptree::point_with_corpus(vp_coords, low, high);
    print_dataset_yav(dist, 1, points_corpus);
    printf("\n");

    points_corpus--;
    uint32_t median = (points_corpus)/2; 
    printf("The median is %d\n", median);

    // exclude vantage point itself
    if(points_corpus!=1) vp_mu[index_node] = Vptree::kselect_dist_corpus_index(dist + 1, _corpus, _indeces, low + 1, points_corpus, median);
    else vp_mu[index_node] = dist[1];

    printf("Rearanged dist\n");
    print_dataset_yav(dist, 1, points_corpus+1);
    printf("The radius/threshold is %lf\n", vp_mu[index_node]);
    printf("Index of the radius threshold %d\n", _indeces[low+1+median]);
    printf("Partition corpus based on distance\n");
    print_dataset_yav_range(_corpus, low, high, _dimensions);
    printf("\n");
    for(uint32_t i = low; i < high; i++) {
        printf("%d ", _indeces[i]);
    }
    printf("\n");

    free(dist);

    //vp->left = makeTree(low + 1, low + 1 + median);
    makeTree(low + 1, low + 1 + median, LEFT_CHILD(index_node)); 

    //vp->right = makeTree(low + 1 + median, high);
    makeTree(low + 1 + median, high, RIGHT_CHILD(index_node));

    return;
}

void Vptree::searchTree(int height, int index_node) {
    if(vp_index[index_node] == FLAG_NO_LEAF) {printf("No leaf\n"); return;}

    double dist = Vptree::points_distance(_target, vp_coords + index_node*_dimensions);

    printf("The height is %d\n", height);
    printf("The index %d and the dist %f and the radius of the vp %f\n", vp_index[index_node], dist, vp_mu[index_node]);
    printf("The furthest distance is %f\n", _tau);
    printf("The vantage point coordinates\n");
    print_dataset_yav(vp_coords + index_node*_dimensions, 1, _dimensions);
    printf("\n");

    std::pair<double, uint32_t> NodeData;

    if (dist < _tau) {
        if (heap.size() == _k) heap.pop();
        NodeData.first = dist;
        NodeData.second = vp_index[index_node];

        heap.push(NodeData);

        if (heap.size() == _k) _tau = heap.top().first;
    }

    // Leaf
    if ((vp_index[LEFT_CHILD(index_node)] == FLAG_NO_LEAF && vp_index[RIGHT_CHILD(index_node)] == FLAG_NO_LEAF) || height == _target_height_tree) {
        printf("No leavess\n");
        return;
    }

    height++;
    if(dist + _tau > vp_mu[index_node]) searchTree(height, LEFT_CHILD(index_node)); 
    if(dist - _tau < vp_mu[index_node]) searchTree(height, RIGHT_CHILD(index_node));
}

void Vptree::searchKNN(double *dist, uint32_t *idx, double *target, uint32_t k) {
    _tau = std::numeric_limits<double>::max();
    _k = MIN(_n, k);
    _target = target; // 1xd

    printf("Traget\n");
    print_dataset_yav(_target, 1 ,_dimensions);

    /* for(auto i = 0; i < _num_nodes_balanced; i++) { */
    /*     printf("%d node\n", i); */
    /*     printf("The radius %f\n", vp_mu[i]); */
    /*     printf("The index %d \n", vp_index[i]); */
    /*     printf("The coords\n"); */
    /*     for (int j = 0; j < _dimensions; j++) printf("%f ", vp_coords[i*_dimensions + j]); */
    /*     printf("\n"); */
    /* } */

    searchTree(1,0);

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
double *Vptree::point_with_corpus(double *query, int low, int high) {
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

double Vptree::points_distance(double *x, double *y) {
    double distance = 0;

    for (uint32_t k = 0; k < _dimensions; k++) distance += (x[k] - y[k]) * (x[k] - y[k]);
    distance = sqrt(distance);

    return distance;
}

void Vptree::swap_row(int index_first, int index_second, double *array, int cols) {
    double *tmp;
    MALLOC(double, tmp, _dimensions);

    memcpy(tmp, _corpus + index_first, sizeof(double)*_dimensions);
    memcpy(_corpus + index_first, _corpus + index_second, sizeof(double)*_dimensions);
    memcpy(_corpus + index_second, tmp, sizeof(double)*_dimensions); 

    free(tmp);
}

double Vptree::kselect_dist_corpus_index(double *dist, double *corpus, uint32_t *indeces, uint32_t low, int64_t len, int64_t k) {
#define SWAPdist(a, b) { tmp1 = dist[a]; dist[a] = dist[b]; dist[b] = tmp1; }
#define SWAPindeces(a, b) { tmp2 = _indeces[a]; _indeces[a] = _indeces[b]; _indeces[b] = tmp2; }
//#define SWAPcorpus(a, b) { \
    memcpy(tmp3, _corpus + a, sizeof(double)*_dimensions); memcpy(_corpus + a, _corpus + b, sizeof(double)*_dimensions); memcpy(_corpus + b, tmp3, sizeof(double)*_dimensions); }

	int64_t i, st;
    double tmp1;
    uint32_t tmp2;
 
	for (st = i = 0; i < len - 1; i++) {
		if (dist[i] > dist[len-1]) continue;
		SWAPdist(i, st);
		SWAPindeces(i + low, st + low);
		//SWAPcorpus((i + low)*_dimensions, (st + low)*_dimensions);
        Vptree::swap_row((i + low)*_dimensions, (st + low)*_dimensions, corpus, _dimensions);
		st++;
	}
 
	SWAPdist(len-1, st);
	SWAPindeces(len-1 + low, st + low);
	//SWAPcorpus((len-1 + low)*_dimensions, (st + low)*_dimensions);
    Vptree::swap_row((len-1 + low)*_dimensions, (st + low)*_dimensions, corpus, _dimensions);
 

	return k == st	?dist[st] 
			:st > k	? kselect_dist_corpus_index(dist, _corpus, _indeces, low, st, k)
				: kselect_dist_corpus_index(dist + st, _corpus + st * _dimensions, _indeces + st, low + st, len - st, k - st);
}
