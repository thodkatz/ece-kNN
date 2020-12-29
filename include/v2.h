#ifndef V2_H
#define V2_H

#include "Vptree.h"
#include <stdint.h>

knnresult kNN_vptree(double *x, Vptree &vpt, uint32_t n, uint32_t m, uint32_t d, uint32_t k);

#endif 
