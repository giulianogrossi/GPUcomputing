#include <iostream>
#include <time.h>

#include "coordinates_dataset.h"
#include "utils/matrix.h"
#include "neural_network.h"
#include "layers/nn_layer.h"
#include "utils/cuda_API_check.h"
#include "utils/bce_cost.h"

int main() {

	srand( time(NULL) );

	// build a dataset
	size_t batch_size = 100;
	size_t number_of_batches = 21;
	CoordinatesDataset dataset = {batch_size, number_of_batches};	
	dataset.summary();
	dataset.fillCoordinatesDataset();
	size_t num_epochs = 1000;

	std::cout << "Model:" << std::endl; 
	
	NeuralNetwork nn(0.01);
	nn.addLayer(new LinearLayer("linear_1", Shape(2, 16)));
	nn.addLayer(new ReLUActivation("relu"));
	nn.addLayer(new LinearLayer("linear_2", Shape(16, 1)));
	nn.addLayer(new SigmoidActivation("sigmoid_output"));

	std::cout << "   - Added layers: [";
	std::cout << nn.layers[0]->name << ", ";
	std::cout << nn.layers[1]->name << ", ";
	std::cout << nn.layers[2]->name << ", ";
	std::cout << nn.layers[3]->name << "]" << std::endl;

	// network training on (n-1) batches
	Matrix Y;
	for (int epoch = 0; epoch < num_epochs; epoch++) {
		float cost = 0.0;

		// loop over batches
		for (int b = 0; b < number_of_batches - 1; b++) {
			Y = nn.forward(dataset.getBatch(b));
			nn.backprop(Y, dataset.getTarget(b));
			cost += BCE(Y, dataset.getTarget(b));
		}

		// print partials
		if (epoch % 100 == 0) {
			std::cout 	<< "Epoch: " << epoch
							<< ", Cost: " << cost / dataset.getNumOfBatches()
							<< std::endl;
		}
	}

	// compute accuracy on the last batch (test)
	Y = nn.forward(dataset.getBatch(number_of_batches-1));
	Y.copyDeviceToHost();

	float accuracy = computeAccuracy(Y, dataset.getTarget(number_of_batches - 1));
	std::cout 	<< "Accuracy: " << accuracy << std::endl;

	return 0;
}
