
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
	
  // TODO

}

int main(int argc, char *argv[]) {

	
	// CPU procedure
	double iStart = seconds();
	long N = THREADS * BLOCKS * TRIALS_PER_THREAD;
	float P_cpu = Gauss_CPU(N,a,b,max);
	double iElaps = seconds() - iStart;
	P_cpu = P_cpu*A;
	printf("CPU elapsed time: %.5f (sec)\n", iElaps);
	printf("CPU estimate of P = %f [error of %f]\n", P_cpu, abs(P_cpu - P_true));

	// GPU procedure
	
	return 0;
}
