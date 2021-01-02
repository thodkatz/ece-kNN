#ifndef V2_H
#define V2_H

#include "Vptree.h"
#include <stdint.h>

knnresult kNN_vptree(Vptree &vpt, double *y, int n, uint32_t m, uint32_t d, uint32_t k);

#endif 
