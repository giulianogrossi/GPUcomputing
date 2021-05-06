

#include "../../utils/MQDB/mqdb.h"
#include "../../utils/common.h"

#define BLOCK_SIZE 16     // block size
#define TEST_CPU 0

/*
 * Kernel for standard (naive) matrix product
 */
__global__ void matProdKernel(mqdb *A, mqdb *B, mqdb *C, int n) {
	// row & col indexes
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// each thread computes an entry of the product matrix
	if ((row < n) && (col < n)) {
		float val = 0;
		for (int k = 0; k < n; k++)
			val += A->elem[row * n + k] * B->elem[k * n + col];
		C->elem[row * n + col] = val;
	}
}

/*
 * Kernel for block sub-matrix product of mqdb
 */
__global__ void mqdbBlockProd(mqdb *A, mqdb *B, mqdb *C, uint sdim, uint d, uint n) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// jump to the right block sub-matrix
	uint  offset = (n+1)*sdim;

	// each thread computes an entry of the product matrix
	if ((row < d) && (col < d)) {
		float val = 0;
		for (int k = 0; k < d; k++)
			val += A->elem[row * n + k + offset] * B->elem[k * n + col + offset];
		C->elem[row * n + col + offset] = val;
	}
}


/*
 * Test on MQDB kernels using Unified Memory
 */
void testKernelsMQDB_unified(uint n, uint k, cudaEvent_t start, cudaEvent_t stop) {

	// matrix instance generation - Unified Memory
	mqdb *A, *B, *C;
	CHECK(cudaMallocManaged(&A, sizeof(mqdb)));
  CHECK(cudaMallocManaged(&A->blkSize, k*sizeof(int)));
  CHECK(cudaMallocManaged(&A->elem, n*n*sizeof(float)));
	
  CHECK(cudaMallocManaged(&B, sizeof(mqdb)));
  CHECK(cudaMallocManaged(&B->blkSize, k*sizeof(int)));
  CHECK(cudaMallocManaged(&B->elem, n*n*sizeof(float)));
	
  CHECK(cudaMallocManaged(&C, sizeof(mqdb)));
  CHECK(cudaMallocManaged(&C->blkSize, k*sizeof(int)));
  CHECK(cudaMallocManaged(&C->elem, n*n*sizeof(float)));

  // random fill mat entries
  int seed = 1;
	genRandDimsAlloc(A, n, k, seed);
	genRandDimsAlloc(B, n, k, seed);
	genRandDimsAlloc(C, n, k, seed);
	fillBlocksAlloc(A, n, k, 'C', 1);
	fillBlocksAlloc(B, n, k, 'C', 2);
	fillBlocksAlloc(C, n, k, 'C', 0);

	ulong nBytes = n * n * sizeof(float);
	printf("Memory size required = %3.4f (MB)\n",(float)nBytes/(1024.0*1024.0));

	/***********************************************************/
	/*                    CPU MQDB product                     */
	/***********************************************************/
	
  printf("CPU MQDB product...\n");
	double CPUtime = 0.0;

  #if TEST_CPU
    double startTm = seconds();
	  mqdbProd(A,B,C);
	  CPUtime = seconds() - startTm;
  #endif

	printf("   CPU elapsed time: %.5f (sec)\n\n", CPUtime);

	/***********************************************************/
	/*                     GPU mat product                     */
	/***********************************************************/
	
  printf("Kernel (naive) mat product...\n");
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
  float milliseconds;
	CHECK(cudaEventRecord(start));
	matProdKernel<<<grid, block>>>(A, B, C, n);
  CHECK(cudaDeviceSynchronize());
	CHECK(cudaEventRecord(stop));
	CHECK(cudaEventSynchronize(stop));
	CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
	float GPUtime1 = milliseconds / 1000.0;
	printf("   elapsed time               : %.2f (sec)\n", GPUtime1);
	printf("   speedup vs CPU MQDB product: %.2f\n\n", CPUtime/GPUtime1);
	//mqdbDisplay(C);

	/***********************************************************/
	/*                     GPU MQDB product                    */
	/***********************************************************/
	
  printf("Kernel MQDB product...\n");
	uint sdim = 0;
	CHECK(cudaEventRecord(start));
	for (uint i = 0; i < k; i++ ) {
		uint d = A->blkSize[i];
		mqdbBlockProd<<<grid, block>>>(A, B, C, sdim, d, n);
		sdim += d;
	}
	CHECK(cudaEventRecord(stop));
	CHECK(cudaEventSynchronize(stop));
	CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
	float GPUtime2 = milliseconds / 1000.0;
	printf("   elapsed time                  :  %.2f (sec)\n", GPUtime2);
	printf("   speedup vs CPU MQDB product   :  %.2f\n", CPUtime/GPUtime2);
	printf("   speedup vs GPU std mat product:  %.2f\n\n", GPUtime1/GPUtime2);

  /***********************************************************/
	/*             GPU MQDB product using streams              */
	/***********************************************************/
	
  printf("GPU MQDB product using streams...\n");

  // create and use streams
	int nstreams = A->nBlocks;
	cudaStream_t streams[nstreams];
	for (int i = 0; i < nstreams; i++)
		CHECK(cudaStreamCreate(&streams[i]));

	uint dsum = 0;  // bound dx
	CHECK(cudaEventRecord(start));
	for (int i = 0; i < nstreams; i++) {
		uint d = A->blkSize[i];
		dim3 grid((d + block.x - 1) / block.x, (d + block.y - 1) / block.y);
		mqdbBlockProd<<<grid, block, 0, streams[i]>>>(A, B, C, dsum, d, n);
		dsum += d;
	}
	CHECK(cudaEventRecord(stop));
	CHECK(cudaEventSynchronize(stop));
	CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
	float GPUtime3 = milliseconds / 1000.0;
	printf("   elapsed time                  : %.5f (sec)\n", GPUtime3);
	printf("   speedup vs CPU MQDB product   : %.2f\n", CPUtime/GPUtime3);
	printf("   speedup vs GPU std mat product: %.2f\n",GPUtime1/GPUtime3);
  printf("   speedup vs GPU MQDB product   : %.2f\n",GPUtime2/GPUtime3);
  //mqdbDisplay(C);

	// clean up streams and events
	for (int i = 0; i < nstreams; i++)
		cudaStreamDestroy(streams[i]);

}

/*
 * main function
 */
int main(int argc, char *argv[]) {
  
  // set up device
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("%s starting mqdb product at ", argv[0]);
	printf("device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	// events to measure time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	uint n = 16*1024;         // matrix size
	uint min_k = 20;       // max num of blocks
	uint max_k = 30;       // max num of blocks

	// multiple tests for k = # diag blocks
	for (uint k = min_k; k <= max_k; k+=5) {
		printf("\n*****   k = %d --- (avg block size = %f)\n",k,(float)n/k);
		testKernelsMQDB_unified(n, k, start, stop);
	}

  cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return 0;
}


