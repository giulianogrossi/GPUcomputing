
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "../../utils/common.h"

#define TRIALS_PER_THREAD 10000
#define BLOCKS  264
#define THREADS 264
#define PI 3.1415926535 // known value of pi

float Gauss_CPU(long trials, float a, float b, float max) {
	long s = 0;
	for (long i = 0; i < trials; i++) {
		float x = (b-a)*(rand() / (float) RAND_MAX)+a;
		float y = (rand() / (float) RAND_MAX);
		s += (y <= expf(-x*x/2));
	}
	return s / (float)trials;
}

__global__ void Gauss_GPU(float *estimate, curandState *states, float a, float b, float max) {
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int s = 0;
	curand_init(tid, 0, 0, &states[tid]);
	for (int i = 0; i < TRIALS_PER_THREAD; i++) {
		float x = (b-a)*curand_uniform(&states[tid])+a;
		float y = curand_uniform(&states[tid]);  // max* dropped
		s += (y <= expf(-x*x/2));
	}
	estimate[tid] = s / (float) TRIALS_PER_THREAD;
}

int main(int argc, char *argv[]) {

	float host[BLOCKS * THREADS];
	float *dev;
	float a = -1;
	float b = 2;
	float max = 1.0f/sqrt(2*PI);
	float A = (b-a)*max;
	float P_true = 0.818594;

	// events to measure time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// CPU procedure
	double iStart = seconds();
	long N = THREADS * BLOCKS * TRIALS_PER_THREAD;
	float P_cpu = Gauss_CPU(N,a,b,max);
	double iElaps = seconds() - iStart;
	P_cpu = P_cpu*A;
	printf("CPU elapsed time: %.5f (sec)\n", iElaps);
	printf("CPU estimate of P = %f [error of %f]\n", P_cpu, abs(P_cpu - P_true));

	// GPU procedure
	curandState *devStates;
	cudaMalloc((void **) &dev, BLOCKS * THREADS * sizeof(float));
	cudaMalloc((void **) &devStates, BLOCKS * THREADS * sizeof(curandState));
	cudaEventRecord(start);
	Gauss_GPU<<<BLOCKS, THREADS>>>(dev, devStates, a, b, max);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaMemcpy(host, dev, BLOCKS * THREADS * sizeof(float), cudaMemcpyDeviceToHost);
	float P = 0.0;
	for (int i = 0; i < BLOCKS * THREADS; i++) {
		P += host[i];
	}
	P = P/(BLOCKS * THREADS)*A;
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	float seconds = milliseconds / 1000.0;
	printf("GPU elapsed time: %.5f (sec)\n", seconds);
	printf("GPU estimate of P = %f [error of %f ]\n", P, abs(P - P_true));
	printf("Speedup = %f\n", iElaps/seconds);
	cudaFree(dev);
	cudaFree(devStates);
	return 0;
}
