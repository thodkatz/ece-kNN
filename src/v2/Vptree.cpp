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

    /* printf("Entering init before search\n"); */
    /* print_dataset_yav(vp_mu, 1, _num_nodes_balanced); */
    /* print_dataset_yav(vp_coords, _num_nodes_balanced, _dimensions); */
    /* print_indeces(vp_index, 1, _num_nodes_balanced); */
    /* printf("\n"); */

    /* printf("The height of the complete tree is %d\n", height_tree); */

    _target_height_tree = _height_tree * target_height_percent;
    assert(0 < _target_height_tree && _target_height_tree <= _height_tree);
    /* printf("The height percentage is %f\n", target_height_percent); */
    /* printf("The requested (rounded) height of the tree is %d\n", _target_height_tree); */

    _sub_nodes_balanced = 0;
    int diff_height = _height_tree - _target_height_tree;
    for(int i = 1; i <= diff_height; i++) _sub_nodes_balanced += pow(2,i);
    /* printf("Sub nodes %d\n", _sub_nodes_balanced); */

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

    /* printf("The dim are: %d\n", _dimensions); */
    /* print_dataset_yav(_corpus, n, _dimensions); */
    /* print_indeces(_indeces, 1, n); */
    /* printf("\n"); */

    _target_height_tree = _height_tree * target_height_tree_percent;
    assert(0 < _target_height_tree && _target_height_tree <= _height_tree);
    printf("The height of the complete tree is %d\n", _height_tree);
    printf("The height percentage is %f\n", target_height_tree_percent);
    printf("The requested (rounded) height of the tree is %d\n", _target_height_tree);
    printf("The total number of nodes for a balanced tree %d\n", _num_nodes_balanced);

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
    printf("Sub nodes %d\n", _sub_nodes_balanced);

    _count_nodes_before_target = pow(2,_target_height_tree-1);
    assert(_target_height_tree-1>=0);

    Vptree::makeTree(1, 0, n, 0);

    /* printf("Final indeces and corpus\n"); */
    /* print_dataset_yav(_corpus, n, _dimensions); */
    /* print_indeces(_indeces, 1, n); */
    /* printf("\n"); */

    /* printf("Final indeces and corpus serialized tree\n"); */
    /* print_dataset_yav(vp_coords, _num_nodes_balanced, _dimensions); */
    /* print_indeces(_indeces, 1, n); */
    /* printf("\n"); */

    /* printf("Final radius\n"); */
    /* print_dataset_yav(vp_mu, 1, _num_nodes_balanced); */
}

void Vptree::makeTree(int height, uint32_t low, uint32_t high, int index_node) {
//#define SWAPcoords(a, b) { \
    memcpy(tmp, _corpus + a, sizeof(double)*_dimensions); memcpy(_corpus + a, _corpus + b, sizeof(double)*_dimensions); memcpy(_corpus + b, tmp, sizeof(double)*_dimensions); }
#define SWAPindeces(a, b) { tmp = _indeces[a]; _indeces[a] = _indeces[b]; _indeces[b] = tmp; }

    /* printf("\nLets make a tree (Christmas?)\n"); */
    /* printf("The low is %d and the high is %d and current height %d, target height %d\n", low, high, height, _target_height_tree); */

    int points_corpus = high - low;
    /* printf("The points are: %d\n", points_corpus); */

    if(points_corpus == 0) return; 
    
    /* printf("The indices are:\n"); */
    /* print_indeces(_indeces + low, 1, high-low); */
    /* printf("\n"); */

    /* printf("The dataset is\n"); */
    if(points_corpus < 0) {printf("Negative pointsss\n"); exit(1);}
    /* print_dataset_yav(_corpus + low*_dimensions, high-low, _dimensions); */

    if(points_corpus == 1) {
        /* printf("We have a leaf!!\n"); */
        memcpy(vp_coords + index_node*_dimensions, _corpus + low*_dimensions, sizeof(double)*_dimensions);
        /* print_dataset_yav(vp_coords + index_node*_dimensions, 1, _dimensions); */
        vp_index[index_node] = _indeces[low];

        /* printf("The vantage point corresponds to the %d element of the global corpus\n", vp_index[index_node]); */
        return;
    }

    static int index_offset = 0;
    // recalibrate index node when stoping tree creation sooner
    if(height == _target_height_tree && height != _height_tree) {
        index_node += _sub_nodes_balanced*index_offset ;
        /* printf("New index %d\n", index_node); */
        index_offset++;
    }

    srand(1);

    // vantage point selection

    //int temp = low; // always take the first
    int temp = rand()%points_corpus + low; // random picked vantage point
    //int temp = select_vp(low, high, index_node); // carefully select vantage point comparing the previous with the current candidate
    //int temp = variance_select_vp(low, high); // carefully select vantage point comparing variance

    /* printf("\nThe random number is %d and index: %d\n", temp, _indeces[temp]); */
    
    // the dist now will have the first element the vantage point
    Vptree::swap_row(temp*_dimensions, low*_dimensions, _corpus, _dimensions);

    /* printf("Swapping distance\n"); */
    if(points_corpus < 0) {printf("Negative corpus points\n"); exit(1);}
    /* print_dataset_yav(_corpus + low*_dimensions, high - low, _dimensions); */
    /* printf("\n"); */
    memcpy(vp_coords + index_node*_dimensions, _corpus + low*_dimensions, sizeof(double) * _dimensions);
    /* printf("Vantage point coordinates and index node %d\n", index_node); */
    /* print_dataset_yav(vp_coords + index_node*_dimensions, 1, _dimensions); */
    /* printf("\n"); */

    /* printf("Swapping indeces\n"); */
    uint32_t tmp;
    SWAPindeces(temp, low);
    /* print_indeces(_indeces + low, 1, high-low); */
    /* printf("\n"); */
    vp_index[index_node] = _indeces[low];

    /* printf("The vantage point corresponds to the %d element of the global corpus\n", vp_index[index_node]); */

    double *dist;
    dist = Vptree::point_with_corpus(vp_coords + index_node*_dimensions, _corpus, low, high);
    //dist = euclidean_distance(_corpus + low*_dimensions, vp_coords + index_node*_dimensions, high-low, _dimensions, 1);
    /* print_dataset_yav(dist, 1, points_corpus); */
    /* printf("\n"); */

    points_corpus--;
    uint32_t median = (points_corpus)/2; // if odd, pick the right 
    /* printf("The median is %d\n", median); */

    // exclude vantage point itself
    if(points_corpus!=1) vp_mu[index_node] = Vptree::kselect_dist_corpus_index(dist + 1, _corpus, _indeces, low + 1, points_corpus, median);
    else vp_mu[index_node] = dist[1];

    /* printf("Rearanged dist\n"); */
    /* print_dataset_yav(dist, 1, points_corpus+1); */
    /* printf("The radius/threshold is %lf\n", vp_mu[index_node]); */
    /* printf("Index of the radius threshold %d\n", _indeces[low+1+median]); */
    /* printf("Partition corpus based on distance\n"); */
    /* print_dataset_yav(_corpus + low*_dimensions, high-low, _dimensions); */
    /* printf("\n"); */
    /* print_indeces(_indeces + low, 1, high-low); */
    /* printf("\n"); */

    free(dist);

    if(height == _target_height_tree && height != _height_tree) {
#define LOG2(n) log(n)/log(2)

        /* printf("Reached the target. Stopping %d of %d\n", _target_height_tree, _height_tree); */
        /* print_dataset_yav(vp_coords, _num_nodes_balanced, _dimensions); */
        /* print_indeces((uint32_t*)vp_index, 1, _num_nodes_balanced); */

        int count = 1;

        /* printf("Max possible size to store leaf values %d\n", _sub_nodes_balanced); */

        // fill the remaining
        for(int i = low+1; i < high; i++) {
            memcpy(vp_coords + (index_node+count)*_dimensions, _corpus + (i)*_dimensions, sizeof(double)*_dimensions);
            vp_index[index_node+count] = _indeces[i];
            count++;
        } 
        /* printf(RED "After\n" RESET); */
        /* print_dataset_yav(vp_coords, _num_nodes_balanced, _dimensions); */
        /* print_indeces((uint32_t*)vp_index, 1, _num_nodes_balanced); */

        return;
    }

    height++;

    makeTree(height, low + 1, low + 1 + median, LEFT_CHILD(index_node)); 

    makeTree(height, low + 1 + median, high, RIGHT_CHILD(index_node));

    return;
}

void Vptree::searchTree(int current_height, int index_node) {
    if(vp_index[index_node] == FLAG_NO_LEAF && current_height == _height_tree) return;
    /* printf("\nIndex node %d\n", index_node); */

    if(current_height == _target_height_tree && current_height != _height_tree) {
        /* printf(RED "Entered target control\n" RESET); */
        // calibrate index

        // which node I am starting from zero in the height before reaching target?
        int offset = index_node + 1 - _count_nodes_before_target;
        if(offset<0) offset=0;
        /* printf("Index node init %d. Offset %d\n", index_node, offset); */

        index_node += _sub_nodes_balanced*offset;
        /* printf("New index node %d\n", index_node); */
        /* printf(RED "Finished target control\n" RESET); */

    }

    double dist = Vptree::points_distance(_target, vp_coords + index_node*_dimensions);

    /* printf("The height is %d\n", current_height); */
    /* printf("The vantage point coordinates\n"); */
    /* print_dataset_yav(vp_coords + index_node*_dimensions, 1, _dimensions); */

    /* printf("Target\n"); */
    /* print_dataset_yav(_target, 1, _dimensions); */

    // by default a priority queue of pairs is ordered by the first element
    std::pair<double, uint32_t> NodeData;

    if (dist < _tau) {
        /* printf("New item pushed to heap with dist %f and index %d\n", dist, vp_index[index_node]); */
        if (heap.size() == _k) heap.pop();

        NodeData.first = dist;
        NodeData.second = vp_index[index_node];

        heap.push(NodeData);

        if (heap.size() == _k) _tau = heap.top().first;
    }
    /* printf("The index %d, the furthest distance %f, the dist %f and the radius of the vp %f\n", vp_index[index_node], _tau, dist, vp_mu[index_node]); */

    if(current_height == _target_height_tree && current_height != _height_tree) {
        /* printf(RED "Reached the target. Stopping %d of %d\n" RESET, _target_height_tree, _height_tree); */

        /* printf("Max possible size to store leaf values %d\n", _sub_nodes_balanced); */

        int i = 1;
        while(vp_index[index_node + i] != FLAG_NO_LEAF && i <= _sub_nodes_balanced) {
            /* printf("The i %d\n", i); */
            double dist = Vptree::points_distance(_target, vp_coords + (index_node + i)*_dimensions);

            /* printf("Vantage point\n"); */
            /* print_dataset_yav(vp_coords + (index_node + i)*_dimensions, 1, _dimensions); */
            /* printf("The index %d, the furthest distance %f, the dist %f and the radius of the vp %f\n", vp_index[index_node + i], _tau, dist, vp_mu[index_node + i]); */
            // should I stick to the priority queue for this comparison?
            if (dist < _tau) {
                /* printf("New item pushed to heap with dist %f and index %d\n\n", dist, vp_index[index_node+i]); */
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
    /* printf("The count is %d\n", _count_nodes); */
    /* printf("\n"); */
    if(dist + _tau > vp_mu[index_node]) searchTree(current_height, RIGHT_CHILD(index_node)); 
    if(dist - _tau < vp_mu[index_node]) searchTree(current_height, LEFT_CHILD(index_node));
}

void Vptree::searchKNN(double *dist, uint32_t *idx, double *target, uint32_t k) {
    _tau = INFINITY;

    _k = MIN(_n, k);
    _target = target; // 1xd

    /* printf("Target\n"); */
    /* print_dataset_yav(_target, 1 ,_dimensions); */

    /* for(auto i = 0; i < _num_nodes_balanced; i++) { */
    /*     printf("%d node\n", i); */
    /*     printf("The radius %f\n", vp_mu[i]); */
    /*     printf("The index %d \n", vp_index[i]); */
    /*     printf("The coords\n"); */
    /*     for (int j = 0; j < _dimensions; j++) printf("%f ", vp_coords[i*_dimensions + j]); */
    /*     printf("\n"); */
    /* } */

    _count_nodes = 1;
    searchTree(1,0);

    /* printf("Nodes visited in the tree %d of total %d\n", _count_nodes, _n); */

    total_nodes_visited += _count_nodes;

    /* printf("The size of the heap is %d\n", heap.size()); */
    int isFirst = 1;
    int count = _k-1;
    while (!heap.empty()) {
        dist[count] = heap.top().first;
        idx[count] = heap.top().second + isFirst;

        count--;
        heap.pop();
    }
    /* printf("The size of the heap is after popping %d\n", heap.size()); */

}

int Vptree::select_vp(int low, int high, int index_node) {
    /* printf(RED "START selection\n" RESET); */
#define LOG2(n) log(n)/log(2)
    if(low == 0 && high == _n) return variance_select_vp(0, _n);

    int len = high - low;
    /* printf("Low is %d and high is %d\n", low, high); */
    const int random_points = MIN(len, SAMPLE_SIZE);

    double *candidates_vp_coords;
    uint32_t *candidates_vp_indeces;
    MALLOC(double, candidates_vp_coords, random_points * _dimensions);
    MALLOC(uint32_t, candidates_vp_indeces, random_points);
    Vptree::sample_and_indeces(candidates_vp_coords, candidates_vp_indeces, random_points, low, high);

    /* printf("Total candidates\n"); */
    /* print_dataset_yav(candidates_vp_coords, random_points, _dimensions); */
    /* print_indeces(candidates_vp_indeces, 1, random_points); */

    int best_index = 0;
    double best_dist = 0;

    // pick the next vp that is furthest from the previous selected ones
    // actually pick the next vp that is furthest from the previous parent

    /* printf("Index is %d\n", index_node); */
    int index_node_parent = (index_node%2==0 ? index_node-=2 : index_node-=1)/2;
    /* printf("Index node parent %d\n", index_node_parent); */
    for(int i = 0; i < random_points; i++) {
        /* printf("\nCandidate for vp\n"); */
        /* print_dataset_yav(candidates_vp_coords + i*_dimensions, 1, _dimensions); */
        /* printf("VP coords\n"); */
        /* print_dataset_yav(vp_coords + index_node_parent*_dimensions, 1, _dimensions); */

        double dist = 0;
        dist += points_distance(vp_coords + index_node_parent*_dimensions, candidates_vp_coords + i*_dimensions);
        /* printf("The dist %f\n", dist); */
        if (dist > best_dist) {
            best_dist = dist;
            best_index = candidates_vp_indeces[i];
        }
    }
    /* printf("\nThe best dist %f\n", best_dist); */
    /* printf("The best index %d\n\n", best_index); */

    /* printf(RED "END selection\n" RESET); */

    free(candidates_vp_coords);
    free(candidates_vp_indeces);

    return best_index;
}

int Vptree::variance_select_vp(int low, int high) {
    /* printf(RED "START selection\n" RESET); */
    const int random_points = MIN(high-low,SAMPLE_SIZE);
    
    double *candidates_vp_coords;
    uint32_t *candidates_vp_indeces;
    MALLOC(double, candidates_vp_coords, random_points * _dimensions);
    MALLOC(uint32_t, candidates_vp_indeces, random_points);
    Vptree::sample_and_indeces(candidates_vp_coords, candidates_vp_indeces, random_points, low, high);

    /* print_dataset_yav(candidates_vp_coords, random_points, _dimensions); */
    /* print_indeces(candidates_vp_indeces, 1, random_points); */

    double *random_test_set;
    MALLOC(double, random_test_set, random_points * _dimensions);
    Vptree::sample(random_test_set, random_points, low, high);

    /* printf("The random test set\n"); */
    /* print_dataset_yav(random_test_set, random_points, _dimensions); */
    /* print_indeces(candidates_vp_indeces, 1, random_points); */

    int best_spread = 0;
    int best_index = 0;
    for(int i = 0; i < random_points; i++) {

        /* printf("The target\n"); */
        /* print_dataset_yav(candidates_vp_coords + i*_dimensions, 1, _dimensions); */
        double *dist = Vptree::point_with_corpus(candidates_vp_coords + i*_dimensions, random_test_set, 0, random_points);
        /* printf("The dist\n"); */
        /* print_dataset_yav(dist, 1, random_points); */
        double median;
        median = qselect(dist, random_points, random_points/2);
        /* print_dataset_yav(dist, 1, random_points); */
        /* printf("The median is %f\n", median); */

        int spread = 0;
        for(int j = 0; j < random_points; j++) {
            spread += (dist[j] - median)*(dist[j] - median);
        }
        if(spread > best_spread) {
            best_spread = spread;
            best_index = candidates_vp_indeces[i];
        }
        /* printf("The spread is %d\n", spread); */

        free(dist);
    }
    /* printf("The best spread is %d\n", best_spread); */
    /* printf("The best index is %d\n", best_index); */

    free(random_test_set);
    free(candidates_vp_coords);
    free(candidates_vp_indeces);
 
    /* printf(RED "END selection\n" RESET); */
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
