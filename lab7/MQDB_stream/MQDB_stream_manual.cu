

#include "../../utils/MQDB/mqdb.h"
#include "../../utils/common.h"

#define BLOCK_SIZE 16     // block size

/*
 * Kernel for block sub-matrix product of mqdb
 */
__global__ void mqdbBlockProd(mqdb A, mqdb B, mqdb C, uint sdim, uint d, uint n) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// jump to the right block sub-matrix
	uint  offset = (n+1)*sdim;

	// each thread computes an entry of the product matrix
	if ((row < d) && (col < d)) {
		float val = 0;
		for (int k = 0; k < d; k++)
			val += A.elem[row * n + k + offset] * B.elem[k * n + col + offset];
		C.elem[row * n + col + offset] = val;
	}
}

/*
 * Test on MQDB kernels using manual async memory
 */
void testKernelsMQDB_manual_mem(uint n, uint k, cudaEvent_t start, cudaEvent_t stop) {

	// matrices
	mqdb *A, *B, *C;         // host 
	mqdb d_A, d_B, d_C;      // device

  ulong nBytes = n * n * sizeof(float);
  int kBytes = k * sizeof(int);
	printf("Memory size required = %3.4f (MB)\n",(float)nBytes/(1024.0*1024.0));


  // host and device Memory
	CHECK(cudaMallocHost(&A, sizeof(mqdb)));
  CHECK(cudaMallocHost(&A->blkSize, kBytes));
  CHECK(cudaMallocHost(&A->elem, nBytes));
  CHECK(cudaMalloc(&d_A.blkSize, kBytes));
  CHECK(cudaMalloc(&d_A.elem, nBytes));

  CHECK(cudaMallocHost(&B, sizeof(mqdb)));
  CHECK(cudaMallocHost(&B->blkSize, kBytes));
  CHECK(cudaMallocHost(&B->elem, nBytes));
	CHECK(cudaMalloc(&d_B.blkSize, kBytes));
  CHECK(cudaMalloc(&d_B.elem, nBytes));
	
  CHECK(cudaMallocHost(&C, sizeof(mqdb)));
  CHECK(cudaMallocHost(&C->blkSize, kBytes));
  CHECK(cudaMallocHost(&C->elem, nBytes));
	CHECK(cudaMalloc(&d_C.blkSize, kBytes));
  CHECK(cudaMalloc(&d_C.elem, nBytes));
  printf("OKKIO\n");

  // random fill mat entries
  int seed = 1;
	genRandDimsAlloc(A, n, k, seed);
	genRandDimsAlloc(B, n, k, seed);
	genRandDimsAlloc(C, n, k, seed);
	fillBlocksAlloc(A, n, k, 'C', 1);
	fillBlocksAlloc(B, n, k, 'C', 2);
	fillBlocksAlloc(C, n, k, 'C', 0);

	// copy blk sizes on device memory
	//CHECK(cudaMemcpy(d_A->blkSize, A->blkSize, kBytes, cudaMemcpyHostToDevice));
	//CHECK(cudaMemcpy(d_B->blkSize, B->blkSize, kBytes, cudaMemcpyHostToDevice));
	//CHECK(cudaMemcpy(d_C->blkSize, C->blkSize, kBytes, cudaMemcpyHostToDevice));
	
  /***********************************************************/
	/*       GPU MQDB product using streams & async copy       */
	/***********************************************************/
	printf("GPU MQDB product using streams...\n");

	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  int nstreams = A->nBlocks;
	cudaStream_t streams[nstreams];
	uint dsum = 0;  // bound dx
	CHECK(cudaEventRecord(start));
	for (int i = 0; i < nstreams; i++) {
		CHECK(cudaStreamCreate(&streams[i]));
		uint d = A->blkSize[i];
    int offset = dsum*n;
    int streamBytes = d*n;
		CHECK(cudaMemcpyAsync(&d_A.elem[offset], &A->elem[offset], streamBytes, cudaMemcpyHostToDevice, streams[i]));
    CHECK(cudaMemcpyAsync(&d_B.elem[offset], &B->elem[offset], streamBytes, cudaMemcpyHostToDevice, streams[i]));
		dim3 grid((d + block.x - 1) / block.x, (d + block.y - 1) / block.y);
		mqdbBlockProd<<<grid, block, 0, streams[i]>>>(d_A, d_B, d_C, dsum, d, n);
    CHECK(cudaMemcpyAsync(&C->elem[offset], &d_C.elem[offset], streamBytes, cudaMemcpyDeviceToHost, streams[i]));
		dsum += d;
	}
	CHECK(cudaEventRecord(stop));
	CHECK(cudaEventSynchronize(stop));
  float milliseconds;
	CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
	float GPUtime3 = milliseconds / 1000.0;
	printf("   elapsed time                  : %.5f (sec)\n", GPUtime3);

	// clean up streams and events
	for (int i = 0; i < nstreams; i++)
		cudaStreamDestroy(streams[i]);

  CHECK(cudaFreeHost(A->elem));
	CHECK(cudaFreeHost(B->elem));
	CHECK(cudaFreeHost(C->elem));
  CHECK(cudaFreeHost(A->blkSize));
	CHECK(cudaFreeHost(B->blkSize));
	CHECK(cudaFreeHost(C->blkSize));
  CHECK(cudaFreeHost(A));
	CHECK(cudaFreeHost(B));
	CHECK(cudaFreeHost(C));
	CHECK(cudaFree(d_A.elem));
	CHECK(cudaFree(d_B.elem));
	CHECK(cudaFree(d_C.elem));
  CHECK(cudaFree(d_A.blkSize));
	CHECK(cudaFree(d_B.blkSize));
	CHECK(cudaFree(d_C.blkSize));
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
		testKernelsMQDB_manual_mem(n, k, start, stop);
	}

  cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return 0;
}



