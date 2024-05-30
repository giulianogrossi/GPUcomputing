#pragma once

#include <iostream>
#include "../utils/matrix.h"

struct NNLayer {
	std::string name;

	// methods
	virtual Matrix& forward(Matrix& ) = 0;
	virtual Matrix& backprop(Matrix& dZ, float learning_rate) = 0;
	std::string getName() { return name; };
};

struct LinearLayer : NNLayer {
	Shape shape;
	const float weights_init_threshold = 0.01;

	// params
	Matrix W;     
	Matrix b;     

	// data
	Matrix Z;    
	Matrix A;    
	Matrix dA;

	// constructor 
	LinearLayer(std::string name, Shape);

	// methods
	void initializeBiasWithZeros();
	void initializeWeightsRandomly();
	void computeAndStoreBackpropError(Matrix& dZ);
	void computeAndStoreLayerOutput(Matrix& A);
	void updateWeights(Matrix& dZ, float learning_rate);
	void updateBias(Matrix& dZ, float learning_rate);
	Matrix& forward(Matrix& A);
	Matrix& backprop(Matrix& dZ, float learning_rate = 0.01);
	int getXDim() const;
	int getYDim() const;
	Matrix getWeightsMatrix() const;
	Matrix getBiasVector() const;
};

struct ReLUActivation : NNLayer {
	Matrix A;
	Matrix Z;
	Matrix dZ;

	// constructor 
	ReLUActivation(std::string name) {this->name = name;};

	// methods
	Matrix& forward(Matrix& Z);
	Matrix& backprop(Matrix& dA, float learning_rate = 0.01);
};

struct SigmoidActivation : NNLayer {
	Matrix A;
	Matrix Z;
	Matrix dZ;

	// constructor 
	SigmoidActivation(std::string name) {this->name = name;};

	// methods
	Matrix& forward(Matrix& Z);
	Matrix& backprop(Matrix& dA, float learning_rate = 0.01);
};
