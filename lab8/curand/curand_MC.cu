
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "../../utils/common.h"


#define TRIALS_PER_THREAD 10000
#define BLOCKS  264
#define THREADS 264
#define PI 3.1415926535 // known value of pi

float pi_mc_CPU(long trials) {
	long points_in_circle = 0;
	for (long i = 0; i < trials; i++) {
		float x = rand() / (float) RAND_MAX;
		float y = rand() / (float) RAND_MAX;
		points_in_circle += (x * x + y * y <= 1.0f);
	}
	return 4.0f * points_in_circle / trials;
}

__global__ void pi_mc_GPU(float *estimate, curandState *states) {
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int points_in_circle = 0;
	curand_init(tid, 0, 0, &states[tid]);
	for (int i = 0; i < TRIALS_PER_THREAD; i++) {
		float x = curand_uniform(&states[tid]);
		float y = curand_uniform(&states[tid]);
		points_in_circle += (x * x + y * y <= 1.0f);
	}
	estimate[tid] = 4.0f * points_in_circle / (float) TRIALS_PER_THREAD;
}

int main(int argc, char *argv[]) {

	float host[BLOCKS * THREADS];
	float *dev;

	// events to measure time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// CPU procedure
	double iStart = seconds();
	float pi_cpu = pi_mc_CPU(THREADS * BLOCKS * TRIALS_PER_THREAD);
	double iElaps = seconds() - iStart;
	printf("CPU elapsed time: %.5f (sec)\n", iElaps);
	printf("CPU estimate of PI = %f [error of %f]\n", pi_cpu, abs(pi_cpu - PI));

	// GPU procedure
	curandState *devStates;
	cudaMalloc((void **) &dev, BLOCKS * THREADS * sizeof(float));
	cudaMalloc((void **) &devStates, BLOCKS * THREADS * sizeof(curandState));
	cudaEventRecord(start);

	pi_mc_GPU<<<BLOCKS, THREADS>>>(dev, devStates);
	
  cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaMemcpy(host, dev, BLOCKS * THREADS * sizeof(float), cudaMemcpyDeviceToHost);
	float pi_gpu = 0.0;
	for (int i = 0; i < BLOCKS * THREADS; i++)
		pi_gpu += host[i];
	pi_gpu /= (BLOCKS * THREADS);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("\nGPU elapsed time (curand Monte Carlo): %.5f (sec)\n", milliseconds / 1000);
	printf("GPU estimate of PI = %f [error of %f ]\n", pi_gpu, abs(pi_gpu - PI));
  printf("Speed-up           = %.0f\n", iElaps/milliseconds*1000);
	cudaFree(dev);
	cudaFree(devStates);
	return 0;
}

