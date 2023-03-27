#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "/content/GPUcomputing/utils/common.h"


/*
 *  Block by block parallel implementation with divergence (sequential schema)
 */
__global__ void blockParReduce1(int *in, int *out, ulong n) {

	uint tid = threadIdx.x;
	ulong idx = blockIdx.x * blockDim.x + threadIdx.x;

	// boundary check
	if (idx >= n)
		return;

	// convert global data pointer to the local pointer of this block
	int *thisBlock = in + blockIdx.x * blockDim.x;

	// in-place reduction in global memory
	for (int stride = 1; stride < blockDim.x; stride *= 2) {
		if ((tid % (2 * stride)) == 0)
			thisBlock[tid] += thisBlock[tid + stride];

		// synchronize within threadblock
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0)
		out[blockIdx.x] = thisBlock[0];
}

/*
 *  Block by block parallel implementation without divergence (interleaved schema)
 */
__global__ void blockParReduce2(int *in, int *out, ulong n) {

	uint tid = threadIdx.x;
	ulong idx = blockIdx.x * blockDim.x + threadIdx.x;

	// boundary check
	if (idx >= n)
		return;

	// convert global data pointer to the local pointer of this block
	int *thisBlock = in + blockIdx.x * blockDim.x;

	// in-place reduction in global memory
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)  {
		if (tid < stride)
			thisBlock[tid] += thisBlock[tid + stride];

		// synchronize within threadblock
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0)
		out[blockIdx.x] = thisBlock[0];
}


/*
 * MAIN: test on parallel reduction
 */
int main(void) {
	int *a, *b, *d_a, *d_b;
	int blockSize = 1024;            // block dim 1D
	ulong numBlock = 1024*1024;      // grid dim 1D
	ulong n = blockSize * numBlock;  // array dim
	long sum_CPU = 0, sum_GPU;
	long nByte = n*sizeof(int), mByte = numBlock * sizeof(int);
	double start, stopGPU, stopCPU, speedup;

	printf("\n****  test on parallel reduction  ****\n");

	// init
	a = (int *) malloc(nByte);
	b = (int *) malloc(mByte);
	for (ulong i = 0; i < n; i++) a[i] = 1;

	CHECK(cudaMalloc((void **) &d_a, nByte));
	CHECK(cudaMemcpy(d_a, a, nByte, cudaMemcpyHostToDevice));
	CHECK(cudaMalloc((void **) &d_b, mByte));
	CHECK(cudaMemset((void *) d_b, 0, mByte));

	/***********************************************************/
	/*                     CPU reduction                       */
	/***********************************************************/
	printf("  Vector length: %.2f MB\n",n/(1024.0*1024.0));
	printf("\n  CPU procedure...\n");
	start = seconds();
	for (ulong i = 0; i < n; i++) 
    sum_CPU += a[i];
	stopCPU = seconds() - start;
	printf("    Elapsed time: %f (sec) \n", stopCPU);
	printf("    sum: %lu\n",sum_CPU);

	printf("\n  GPU kernels (mem required %lu bytes)\n", nByte);

	/***********************************************************/
	/*         KERNEL blockParReduce1 (divergent)              */
	/***********************************************************/
	// block by block parallel implementation with divergence
	printf("\n  Launch kernel: blockParReduce1...\n");
	start = seconds();
	blockParReduce1<<<numBlock, blockSize>>>(d_a, d_b, n);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());
	stopGPU = seconds() - start;
	speedup = stopCPU/stopGPU;
	printf("    Elapsed time: %f (sec) - speedup %.1f\n", stopGPU,speedup);
	
  // memcopy D2H
	CHECK(cudaMemcpy(b, d_b, mByte, cudaMemcpyDeviceToHost));
	
  // check result
	sum_GPU = 0;
	for (uint i = 0; i < numBlock; i++)
		sum_GPU += b[i];
	assert(sum_GPU == n);

	// reset input vector on GPU
	for (ulong i = 0; i < n; i++) a[i]=1;
	CHECK(cudaMemcpy(d_a, a, nByte, cudaMemcpyHostToDevice));

	/***********************************************************/
	/*        KERNEL blockParReduce2  (non divergent)          */
	/***********************************************************/
	// block by block parallel implementation without divergence
	printf("\n  Launch kernel: blockParReduce2...\n");
	start = seconds();
	blockParReduce2<<<numBlock, blockSize>>>(d_a, d_b, n);
	CHECK(cudaDeviceSynchronize());
	stopGPU = seconds() - start;
	speedup = stopCPU/stopGPU;
	printf("    Elapsed time: %f (sec) - speedup %.1f\n", stopGPU,speedup);
	CHECK(cudaGetLastError());
	
  // memcopy D2H
	CHECK(cudaMemcpy(b, d_b, mByte, cudaMemcpyDeviceToHost));
	
  // check result
	sum_GPU = 0;
	for (uint i = 0; i < numBlock; i++) {
		sum_GPU += b[i];
  //		printf("b[%d] = %d\n",i,b[i]);
	}
	assert(sum_GPU == n);
	
  // reset input vector on GPU
	for (ulong i = 0; i < n; i++) a[i] = 1;
	CHECK(cudaMemcpy(d_a, a, nByte, cudaMemcpyHostToDevice));

	// check result
	sum_GPU = 0;
	for (uint i = 0; i < numBlock; i++)
		sum_GPU += b[i];
	assert(sum_GPU == n);

	cudaFree(d_a);

	CHECK(cudaDeviceReset());
	return 0;
}
