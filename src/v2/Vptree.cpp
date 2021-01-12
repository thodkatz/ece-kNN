#include <stdint.h>
#include "Vptree.h"
#include "v0.h"
#include "utils.h"
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
#include <cblas.h>

#define FLAG_NO_LEAF -1
#define LEFT_CHILD(index) index*2 + 1
#define RIGHT_CHILD(index) index*2 + 2
#define SAMPLE_SIZE 5 // the size of the random sample used to evaluate the vantage point

void Vptree::init_before_search(int n, int num_nodes_balanced, int height_tree, float target_height_percent, double *vp_mu, double *vp_coords, int *vp_index) {
    // Everything you need to search the tree without creating explicitly

    _num_nodes_balanced = num_nodes_balanced;
    _height_tree = height_tree;
    _n = n;

    this->vp_mu = vp_mu;
    this->vp_coords = vp_coords;
    this->vp_index = vp_index;

    _target_height_tree = _height_tree * target_height_percent;
    assert(0 < _target_height_tree && _target_height_tree <= _height_tree);

    _sub_nodes_balanced = 0;
    int diff_height = _height_tree - _target_height_tree;
    for(int i = 1; i <= diff_height; i++) _sub_nodes_balanced += pow(2,i);

    _count_nodes_before_target = pow(2,_target_height_tree-1);
    assert(_target_height_tree-1>=0);
}

Vptree::Vptree(double *corpus, uint32_t *indeces, uint32_t n, uint32_t dimensions, int num_nodes_balanced, int height_tree, float target_height_tree_percent) {
    _dimensions = dimensions;
    _corpus = corpus;
    _indeces = indeces;
    _n = n;
    _num_nodes_balanced = num_nodes_balanced;
    _height_tree = height_tree;

    _target_height_tree = _height_tree * target_height_tree_percent;
    assert(0 < _target_height_tree && _target_height_tree <= _height_tree);

    // init each node of the vptree
    MALLOC(double, vp_mu, _num_nodes_balanced);
    MALLOC(double, vp_coords, _num_nodes_balanced * _dimensions);
    MALLOC(int, vp_index, _num_nodes_balanced);
    for(int i = 0; i < _num_nodes_balanced; i++) {
        vp_mu[i] = FLAG_NO_LEAF;
        vp_index[i] = FLAG_NO_LEAF;
        for(int j = 0; j < _dimensions; j++) vp_coords[i*_dimensions+ j] = 0;
    }

    _sub_nodes_balanced = 0;
    int diff_height = _height_tree - _target_height_tree;
    for(int i = 1; i <= diff_height; i++) _sub_nodes_balanced += pow(2,i);

    _count_nodes_before_target = pow(2,_target_height_tree-1);
    assert(_target_height_tree-1>=0);
    /* struct timespec tic; */
    /* struct timespec toc; */
    /* TIC() */    
    Vptree::makeTree(1, 0, n, 0);
    //TOC("Time elasped making tree %lf\n");
}

void Vptree::makeTree(int height, uint32_t low, uint32_t high, int index_node) {
//#define SWAPcoords(a, b) { \
    memcpy(tmp, _corpus + a, sizeof(double)*_dimensions); memcpy(_corpus + a, _corpus + b, sizeof(double)*_dimensions); memcpy(_corpus + b, tmp, sizeof(double)*_dimensions); }
#define SWAPindeces(a, b) { tmp = _indeces[a]; _indeces[a] = _indeces[b]; _indeces[b] = tmp; }

    int points_corpus = high - low;

    if(points_corpus == 0) return; 

    if(points_corpus == 1) {
        memcpy(vp_coords + index_node*_dimensions, _corpus + low*_dimensions, sizeof(double)*_dimensions);
        vp_index[index_node] = _indeces[low];

        return;
    }

    static int index_offset = 0;
    // recalibrate index node when stoping tree creation sooner
    if(height == _target_height_tree && height != _height_tree) {
        index_node += _sub_nodes_balanced*index_offset ;
        index_offset++;
    }

    srand(1);

    // vantage point selection

    //int temp = low; // always take the first
    int temp = rand()%points_corpus + low; // random picked vantage point
    //int temp = select_vp(low, high, index_node); // carefully select vantage point comparing the previous with the current candidate
    //int temp = variance_select_vp(low, high); // carefully select vantage point comparing variance

    // the dist now will have the first element the vantage point
    Vptree::swap_row(temp*_dimensions, low*_dimensions, _corpus, _dimensions);

    if(points_corpus < 0) {printf("Negative corpus points\n"); exit(1);}
    memcpy(vp_coords + index_node*_dimensions, _corpus + low*_dimensions, sizeof(double) * _dimensions);

    uint32_t tmp;
    SWAPindeces(temp, low);
    vp_index[index_node] = _indeces[low];

    double *dist;
    dist = Vptree::point_with_corpus(vp_coords + index_node*_dimensions, _corpus, low, high);

    points_corpus--;
    uint32_t median = (points_corpus)/2; // if odd, pick the right 

    // exclude vantage point itself
    if(points_corpus!=1) vp_mu[index_node] = Vptree::kselect_dist_corpus_index(dist + 1, _corpus, _indeces, low + 1, points_corpus, median);
    else vp_mu[index_node] = dist[1];

    free(dist);

    if(height == _target_height_tree && height != _height_tree) {
#define LOG2(n) log(n)/log(2)

        int count = 1;

        // fill the remaining
        for(int i = low+1; i < high; i++) {
            memcpy(vp_coords + (index_node+count)*_dimensions, _corpus + (i)*_dimensions, sizeof(double)*_dimensions);
            vp_index[index_node+count] = _indeces[i];
            count++;
        } 

        return;
    }

    height++;

    makeTree(height, low + 1, low + 1 + median, LEFT_CHILD(index_node)); 

    makeTree(height, low + 1 + median, high, RIGHT_CHILD(index_node));

    return;
}

void Vptree::searchTree(int current_height, int index_node) {
    if(vp_index[index_node] == FLAG_NO_LEAF && current_height == _height_tree) return;

    if(current_height == _target_height_tree && current_height != _height_tree) {
        // calibrate index

        // which node I am starting from zero in the height before reaching target?
        int offset = index_node + 1 - _count_nodes_before_target;
        if(offset<0) offset=0;
        index_node += _sub_nodes_balanced*offset;

    }

    double dist = Vptree::points_distance(_target, vp_coords + index_node*_dimensions);

    // by default a priority queue of pairs is ordered by the first element
    std::pair<double, uint32_t> NodeData;

    if (dist < _tau) {
        if (heap.size() == _k) heap.pop();

        NodeData.first = dist;
        NodeData.second = vp_index[index_node];

        heap.push(NodeData);

        if (heap.size() == _k) _tau = heap.top().first;
    }

    if(current_height == _target_height_tree && current_height != _height_tree) {
        int i = 1;
        while(vp_index[index_node + i] != FLAG_NO_LEAF && i <= _sub_nodes_balanced) {
            double dist = Vptree::points_distance(_target, vp_coords + (index_node + i)*_dimensions);

            // should I stick to the priority queue for this comparison?
            if (dist < _tau) {
                if (heap.size() == _k) heap.pop();

                NodeData.first = dist;
                NodeData.second = vp_index[index_node + i];

                heap.push(NodeData);

                if (heap.size() == _k) _tau = heap.top().first;
            }

            i++;
        }

        return;
    }

    if(current_height == _height_tree) return;

    // Leaf
    if((vp_index[LEFT_CHILD(index_node)] == FLAG_NO_LEAF && vp_index[RIGHT_CHILD(index_node)] == FLAG_NO_LEAF) && _target_height_tree == _height_tree) return;

    current_height++;
    _count_nodes++;
    if(dist + _tau > vp_mu[index_node]) searchTree(current_height, RIGHT_CHILD(index_node)); 
    if(dist - _tau < vp_mu[index_node]) searchTree(current_height, LEFT_CHILD(index_node));
}

void Vptree::searchKNN(double *dist, uint32_t *idx, double *target, uint32_t k) {
    _tau = INFINITY;

    _k = MIN(_n, k);
    _target = target; // 1xd

    _count_nodes = 1;
    searchTree(1,0);

    /* printf("Nodes visited in the tree %d of total %d\n", _count_nodes, _n); */

    total_nodes_visited += _count_nodes;

    int isFirst = 1;
    int count = _k-1;
    while (!heap.empty()) {
        dist[count] = heap.top().first;
        idx[count] = heap.top().second + isFirst;

        count--;
        heap.pop();
    }

}

int Vptree::select_vp(int low, int high, int index_node) {
#define LOG2(n) log(n)/log(2)
    if(low == 0 && high == _n) return variance_select_vp(0, _n);

    int len = high - low;
    const int random_points = MIN(len, SAMPLE_SIZE);

    double *candidates_vp_coords;
    uint32_t *candidates_vp_indeces;
    MALLOC(double, candidates_vp_coords, random_points * _dimensions);
    MALLOC(uint32_t, candidates_vp_indeces, random_points);
    Vptree::sample_and_indeces(candidates_vp_coords, candidates_vp_indeces, random_points, low, high);

    int best_index = 0;
    double best_dist = 0;

    // pick the next vp that is furthest from the previous selected ones
    // actually pick the next vp that is furthest from the previous parent

    int index_node_parent = (index_node%2==0 ? index_node-=2 : index_node-=1)/2;
    for(int i = 0; i < random_points; i++) {

        double dist = 0;
        dist += points_distance(vp_coords + index_node_parent*_dimensions, candidates_vp_coords + i*_dimensions);
        if (dist > best_dist) {
            best_dist = dist;
            best_index = candidates_vp_indeces[i];
        }
    }

    free(candidates_vp_coords);
    free(candidates_vp_indeces);

    return best_index;
}

int Vptree::variance_select_vp(int low, int high) {
    const int random_points = MIN(high-low,SAMPLE_SIZE);
    
    double *candidates_vp_coords;
    uint32_t *candidates_vp_indeces;
    MALLOC(double, candidates_vp_coords, random_points * _dimensions);
    MALLOC(uint32_t, candidates_vp_indeces, random_points);
    Vptree::sample_and_indeces(candidates_vp_coords, candidates_vp_indeces, random_points, low, high);

    double *random_test_set;
    MALLOC(double, random_test_set, random_points * _dimensions);
    Vptree::sample(random_test_set, random_points, low, high);

    int best_spread = 0;
    int best_index = 0;
    for(int i = 0; i < random_points; i++) {

        double *dist = Vptree::point_with_corpus(candidates_vp_coords + i*_dimensions, random_test_set, 0, random_points);
        double median;
        median = qselect(dist, random_points, random_points/2);

        int spread = 0;
        for(int j = 0; j < random_points; j++) {
            spread += (dist[j] - median)*(dist[j] - median);
        }
        if(spread > best_spread) {
            best_spread = spread;
            best_index = candidates_vp_indeces[i];
        }

        free(dist);
    }

    free(random_test_set);
    free(candidates_vp_coords);
    free(candidates_vp_indeces);
 
    return best_index;
}

// reservoir sampling. Copy random element from corpus to array
void Vptree::sample_and_indeces(double *vals, uint32_t *indeces, int num, int low, int high) {
    int corpus_size = high - low;
    num = MIN(corpus_size, num);

    for(int i = 0; i < num; i ++) {
        memcpy(vals + i*_dimensions, _corpus + (low+i)*_dimensions, sizeof(double) * _dimensions);
        indeces[i] = low + i;
    }

    for (int i = num; i < corpus_size; i++) { 
        int j = rand() % (i+1); 
        if (j < num) {
            memcpy(vals + j*_dimensions, _corpus + (low+i)*_dimensions, sizeof(double) * _dimensions);
            indeces[j] = low + i;
        }
    } 
}

void Vptree::sample(double *vals, int num, int low, int high) {
    int corpus_size = high - low;
    num = MIN(corpus_size, num);

    for(int i = 0; i < num; i ++) {
        memcpy(vals + i*_dimensions, _corpus + (low+i)*_dimensions, sizeof(double) * _dimensions);
    }

    for (int i = num; i < corpus_size; i++) { 
        int j = rand() % (i+1); 
        if (j < num) {
            memcpy(vals + j*_dimensions, _corpus + (low+i)*_dimensions, sizeof(double) * _dimensions);
        }
    } 
}


// naive euclidean distance matrix.
double *Vptree::point_with_corpus(double *query, double *corpus, int low, int high) {
    int points = high - low;
    double *distance;
    MALLOC(double, distance, points);
    int count = 0;

    for(int j = low; j < high; j++) {
        double temp = 0;
        for (int k = 0; k < _dimensions; k++) {
            temp += (corpus[j*_dimensions + k] - query[k]) * (corpus[j*_dimensions + k] - query[k]);
        }
        distance[count] = sqrt(temp);
        count++;
    }

    return distance;
}

double Vptree::points_distance(double *x, double *y) {
    double distance = 0;

    //for (uint32_t k = 0; k < _dimensions; k++) distance += (x[k] - y[k]) * (x[k] - y[k]);
    double d1 = cblas_ddot(_dimensions, x, 1, x, 1);
    double d2 = cblas_ddot(_dimensions, y, 1, y, 1);
    double d3 = cblas_ddot(_dimensions, x, 1, y, 1);
    distance = d1 -2*d3 + d2;   
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
