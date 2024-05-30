#include "coordinates_dataset.h"
#include <iostream>

// Populate the dataset
void CoordinatesDataset::fillCoordinatesDataset() {
	// loop on batches
	for (int i = 0; i < number_of_batches; i++) {
		batches.push_back(Matrix(Shape(batch_size, 2)));
		targets.push_back(Matrix(Shape(batch_size, 1)));

		batches[i].allocateMemory();
		targets[i].allocateMemory();

		// loop on batch elements
		for (int k = 0; k < batch_size; k++) {
			float a = (float)rand() / RAND_MAX - 0.5;
			float b = (float)rand() / RAND_MAX - 0.5;
			batches[i].data_host[k] = a*2.0;
			batches[i].data_host[batch_size + k] = b*2.0;

			// positive and negative classes
			if (a*b > 0) 
				targets[i].data_host[k] = 1;
			else 
				targets[i].data_host[k] = 0;
		}

		// copy to device
		batches[i].copyHostToDevice();
		targets[i].copyHostToDevice();
	}
}

void CoordinatesDataset::summary() {
	std::cout << "CoordinatesDataset:" << std::endl;
	std::cout << "   - batch size: \t" << batch_size << std::endl;
	std::cout << "   - num of batches:\t" << number_of_batches << std::endl;
}

int CoordinatesDataset::getNumOfBatches() {
	return number_of_batches;
}

Matrix CoordinatesDataset::getBatch(int i) {
	return batches[i];
}

Matrix CoordinatesDataset::getTarget(int i) {
	return targets[i];
}
