#ifndef KERNELS_H_
#define KERNELS_H_

#include "ppm.h"
#define BLOCK_SIZE    32
#define MASK_SIZE     21
#define TILE_SIZE     (BLOCK_SIZE + MASK_SIZE - 1)


typedef struct {
   int width;
   int height;
   float* elements;
 } Matrix;

// function declaration
__global__ void equalize(Matrix A, float *histogram);
__global__ void hist(Matrix A, float *histogram);
__global__ void norm_hist(float *histogram);
__global__ void conv2D(Matrix A, Matrix B, Matrix M);

#endif