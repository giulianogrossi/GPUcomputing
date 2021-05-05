
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include "../../utils/common.h"

#define TRIALS_PER_THREAD 10000
#define BLOCKS  264
#define THREADS 264
#define PI 3.1415926535 // known value of pi

int main(void) {
    
	long trials = THREADS * BLOCKS * TRIALS_PER_THREAD; // num points

  printf("Number of random points in the square = %lu\n", trials);

	curandGenerator_t gen;
	float *X_d, *X, *Y_d, *Y ;

	// Allocate points on host
	X = (float *) malloc(trials * sizeof(float));
  Y = (float *) malloc(trials * sizeof(float));

	/* Allocate n floats on device */
	CHECK(cudaMalloc((void **)&X_d, trials * sizeof(float)));
  CHECK(cudaMalloc((void **)&Y_d, trials * sizeof(float)));

	// Create pseudo-random number generator 
	CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

	// Set seed 
	CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

	// Generate 2*n floats on device 
	CHECK_CURAND(curandGenerateUniform(gen, X_d, trials));
  CHECK_CURAND(curandGenerateUniform(gen, Y_d, trials));

	// Copy device memory to host 
	CHECK(cudaMemcpy(X, X_d, trials * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(Y, Y_d, trials * sizeof(float), cudaMemcpyDeviceToHost));

  // num of points within the circle
  ulong points_in_circle = 0;
  for (long i = 0; i < trials; i++) 
		points_in_circle += (X[i] * X[i] + Y[i] * Y[i] <= 1.0f);

  // estimate PI
	float pi = 4.0f * points_in_circle / (float)trials;
  printf("Estimate of PI = %f [error of %f]\n", pi, abs(pi - PI));

	// Cleanup 
	CHECK_CURAND(curandDestroyGenerator(gen));
	CHECK(cudaFree(X_d));
  CHECK(cudaFree(Y_d));
  free(X);
	free(Y);
	return EXIT_SUCCESS;
}
