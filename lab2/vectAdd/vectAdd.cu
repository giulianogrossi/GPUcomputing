#include <stdio.h>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}

/**
 * CUDA Kernel: vector addition
 */
__global__ void vectorAdd(const float *A, const float *B, float *C,
		int numElements) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
		C[i] = A[i] + B[i];
}

/**
 * MAIN
 */
int main(void) {
	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// Print the vector length to be used, and compute its size
	int N = 50000;
	size_t size = N * sizeof(float);
	printf("[Vector addition of %d elements]\n", N);

	// Allocate the host input vector A,B,C
	float *h_A = (float *) malloc(size);
	float *h_B = (float *) malloc(size);
	float *h_C = (float *) malloc(size);

	// Initialize the host input vectors
	for (int i = 0; i < N; ++i) {
		h_A[i] = rand() % 10;
		h_B[i] = rand() % 10;
	}

	// Allocate the device input vector A,B,C
	float *d_A = NULL;
	CHECK(cudaMalloc((void **) &d_A, size));
	float *d_B = NULL;
	CHECK(cudaMalloc((void **) &d_B, size));
	float *d_C = NULL;
	CHECK(cudaMalloc((void **) &d_C, size));

	// Copy the host input vectors A and B in device memory
	printf("Copy input data from the host memory to the CUDA device\n");
	CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

	// Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
	CHECK(cudaGetLastError());
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
				cudaGetErrorString(err));
		exit (EXIT_FAILURE);
	}

	// Copy the device result vector in host memory
	printf("Copy output data from the CUDA device to the host memory\n");
	CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

	// Verify that the result vector is correct
	for (int i = 0; i < N; ++i) {
		if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
			fprintf(stderr, "Result verification failed at element %d!\n", i);
			exit (EXIT_FAILURE);
		}
	}

	printf("Test PASSED\n");

	// Free device global memory
	CHECK(cudaFree(d_A));
	CHECK(cudaFree(d_B));
	CHECK(cudaFree(d_C));

	// Free host memory
	free(h_A);
	free(h_B);
	free(h_C);

	printf("Done\n");
	return 0;
}

