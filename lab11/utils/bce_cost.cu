#include "bce_cost.h"
#include "cuda_API_check.h"

#include <iostream>

__global__ void binaryCrossEntropyCost(float* predictions, float* target, int size, float* cost) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size) {
		float partial_cost = target[index] * __log2f(predictions[index]) + (1.0f - target[index]) * __log2f(1.0f - predictions[index]);
		atomicAdd(cost, -partial_cost / size);
	}
}

__global__ void dBinaryCrossEntropyCost(float* predictions, float* target, float* dY, int size) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size) {
		dY[index] = -1.0 * ( target[index]/predictions[index] - (1 - target[index])/(1 - predictions[index]) );
	}
}

float BCE(Matrix predictions, Matrix target) {
	float cost = 0.0f;
	float *cost_h = &cost; 
	float *cost_d;
	CHECK(cudaMalloc((float**)&cost_d, sizeof(float)));
	CHECK(cudaMemset(cost_d, 0.0f, sizeof(float)));

	dim3 block_size(256);
	dim3 num_of_blocks((predictions.shape.x_dim + block_size.x - 1) / block_size.x);
	binaryCrossEntropyCost<<<num_of_blocks, block_size>>>(predictions.data_device, target.data_device, predictions.shape.x_dim, cost_d);
	CHECK(cudaMemcpy(cost_h, cost_d, sizeof(float), cudaMemcpyDeviceToHost));
	
	cudaFree(cost_d);
	return *cost_h;
}

Matrix dBCE(Matrix predictions, Matrix target, Matrix dY) {
	dim3 block_size(256);
	dim3 num_of_blocks((predictions.shape.x_dim + block_size.x - 1) / block_size.x);
	dBinaryCrossEntropyCost<<<num_of_blocks, block_size>>>(predictions.data_device,
														   target.data_device,
														   dY.data_device,
														   predictions.shape.x_dim);
	return dY;
}

float computeAccuracy(const Matrix& predictions, const Matrix& targets) {
	int m = predictions.shape.x_dim;
	int correct_predictions = 0;

	for (int i = 0; i < m; i++) {
		float prediction = predictions.data_host[i] > 0.5 ? 1 : 0;
		if (prediction == targets.data_host[i]) {
			correct_predictions++;
		}
	}

	return static_cast<float>(correct_predictions) / m;
}