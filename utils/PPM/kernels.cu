#include "kernels.h"

 /*
 * Kernel 1D for mat equalization (in place)
 */
__global__ void equalize(Matrix A, float *histogram) {

   uint x = blockIdx.x * blockDim.x + threadIdx.x;
   
  // pixel out of range
   if (x >= A.width * A.height)
      return;
   
   // equalize
   A.elements[x] = A.elements[x] * 255 * histogram[(color)A.elements[x]];
}


 /*
 * Kernel 1D that computes histogram 
 */
__global__ void hist(Matrix A, float *histogram) {

   uint x = blockIdx.x * blockDim.x + threadIdx.x;
   
  // pixel out of range
   if (x >= A.width * A.height)
      return;
   
   // use atomic
   atomicAdd(&histogram[(color)A.elements[x]], 1);
}

/*
 * Kernel 1D that computes histogram 
 */
 __global__ void norm_hist(float *histogram) {

   uint x = blockIdx.x * blockDim.x + threadIdx.x;
   __shared__ float smem[256];
   
   smem[x] = histogram[x];  			
   __syncthreads();

   // do reduction in shared mem
	for (int stride = 128; stride > 0; stride >>= 1) {
		if (x < stride)
			smem[x] += smem[x + stride];
		__syncthreads();
	}

   // use atomic
   histogram[x] /= smem[0];
}

 /*   
  * Kernel 2D that computes convolution using shared memory
  *   A: input matrix
  *   B: output matrix
  *   M: convolution mask matrix   
 */
__global__ void conv2D(Matrix A, Matrix B, Matrix M) {
   
   int x = blockIdx.x * blockDim.x + threadIdx.x; // Column index of matrix A
   int y = blockIdx.y * blockDim.y + threadIdx.y; // Row index of matrix A

   int tile_size = BLOCK_SIZE + MASK_SIZE - 1;
   int radius = MASK_SIZE / 2;

   // Allocate shared memory
   __shared__ float smem[TILE_SIZE][TILE_SIZE];
   
   // Load data into shared memory
   for (int row = 0; row <= tile_size/blockDim.y; row++) {
      for (int col = 0; col <= tile_size/blockDim.x; col++) {
         int row_data = y + blockDim.y * row - radius;   // input data index row
         int col_data = x + blockDim.x * col - radius;   // input data index column
         int row_smem = threadIdx.y + blockDim.y * row;  // mask index row
         int col_smem = threadIdx.x + blockDim.x * col;  // mask index column

         // Check valid range for smem and data
         if (row_smem < tile_size && col_smem < tile_size) {
            if (row_data >= 0 && row_data < A.height && col_data >= 0 && col_data < A.width) {
               smem[row_smem][col_smem] = A.elements[row_data * A.width + col_data];
            } else {
               smem[row_smem][col_smem] = 0.0f;
            }
         }
      }
   }
   __syncthreads();

   // Apply convolution
   float sum = 0.0f;
   for (int i = 0; i < MASK_SIZE; i++) {
      for (int j = 0; j < MASK_SIZE; j++) {
         int r = threadIdx.y + i; 
         int c = threadIdx.x + j;
         if (r >= 0 && r < tile_size && c >= 0 && c < tile_size) {
            sum += smem[r][c] * M.elements[i * MASK_SIZE + j];
         }
      }
   }
   
   // Write output
   if (y < A.height && x < A.width) {
      B.elements[y * B.width + x] = sum;
   }
}  
