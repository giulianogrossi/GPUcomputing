
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cufft.h>

#include "../../utils/common.h"

#define BATCH 16

/*
 * An example usage of the cuFFT library. This example performs a 1D forward
 * FFT.
 */

int nprints = 30;

/*
 * Create N fake samplings along the function cos(x). These samplings will be
 * stored as single-precision floating-point values.
 */
void generate_fake_samples(int N, float **out) {
	int i;
	float *result = (float *) malloc(sizeof(float) * N);
	double delta = M_PI / 20.0;
	for (i = 0; i < N; i++)
		result[i] = cos(i * delta);
	*out = result;
}

void rect(uint N, float **out) {
	float *r = (float *) calloc(N, sizeof(float));
	for (uint i = 0; i < N/100; ++i) 
    r[i] = 1.0f;
	*out = r;
}

/*
 * Convert a real-valued vector r of length Nto a complex-valued vector.
 */
void real_to_complex(float *r, cufftComplex **complx, int N) {
	int i;
	(*complx) = (cufftComplex *) malloc(sizeof(cufftComplex) * N);

	for (i = 0; i < N; i++) {
		(*complx)[i].x = r[i];
		(*complx)[i].y = 0;
	}
}

int main(int argc, char **argv) {

	int i;
	int N = 1024*1024;
	float *samples;
	cufftHandle plan = 0;
	cufftComplex *dComplexSamples, *complexSamples, *complexFreq;

	// Input Generation
	rect(N, &samples);

  printf("Start computation...\n");
  double start = seconds();
	real_to_complex(samples, &complexSamples, N);
	
  complexFreq = (cufftComplex *) malloc(sizeof(cufftComplex) * N);

	// Setup the cuFFT plan
	CHECK_CUFFT(cufftPlan1d(&plan, N, CUFFT_C2C, 1));

	// Allocate device memory
	CHECK(cudaMalloc((void **)&dComplexSamples, sizeof(cufftComplex) * N));

	// Transfer inputs into device memory
	CHECK(cudaMemcpy(dComplexSamples, complexSamples, sizeof(cufftComplex) * N, cudaMemcpyHostToDevice));

	// Execute a complex-to-complex 1D FFT
	CHECK_CUFFT(cufftExecC2C(plan, dComplexSamples, dComplexSamples, CUFFT_FORWARD));

	// Retrieve the results into host memory
	CHECK(cudaMemcpy(complexFreq, dComplexSamples, sizeof(cufftComplex) * N, cudaMemcpyDeviceToHost));

  double elaps = seconds() - start;

  printf("Elapsed time: %f (sec)\n", elaps);

  // save FFT on a file
  printf("Save on file...\n");
  FILE *filePtr;
  filePtr = fopen("FFTdata.txt","w");
  for (i = 0; i < N; i++) {
    fprintf(filePtr, "%.3g, %.5g\n", complexFreq[i].x, complexFreq[i].y);
  }
 
	free(samples);
	free(complexSamples);
	free(complexFreq);

	CHECK(cudaFree(dComplexSamples));
	CHECK_CUFFT(cufftDestroy(plan));
	return 0;
}
