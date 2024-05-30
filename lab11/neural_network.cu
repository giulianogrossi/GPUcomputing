#include "neural_network.h"
#include "utils/cuda_API_check.h"
#include "utils/bce_cost.h"
#include <iostream>

void NeuralNetwork::addLayer(NNLayer* layer) {
	layers.push_back(layer);
}

Matrix NeuralNetwork::forward(Matrix X) {
	Matrix Z = X;

	for (auto layer : layers) {
		Z = layer->forward(Z);
	}
	Y = Z;
	return Y;
}

void NeuralNetwork::backprop(Matrix predictions, Matrix target) {
	dY.allocateMemoryIfNotAllocated(predictions.shape);
	Matrix error = dBCE(predictions, target, dY);

	for (auto it = this->layers.rbegin(); it != this->layers.rend(); it++) {
		error = (*it)->backprop(error, learning_rate);
	}
}

std::vector<NNLayer*> NeuralNetwork::getLayers() const {
	return layers;
}
