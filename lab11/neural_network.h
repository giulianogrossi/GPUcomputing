#pragma once

#include <vector>
#include "layers/nn_layer.h"

struct NeuralNetwork {
	float learning_rate;
	std::vector<NNLayer*> layers;
	Matrix Y;
	Matrix dY;

	// constructor
	NeuralNetwork(float lr) : learning_rate(lr) {};
	
	// methods
	Matrix forward(Matrix X);
	void backprop(Matrix predictions, Matrix target);
	void addLayer(NNLayer *layer);
	std::vector<NNLayer*> getLayers() const;
};
