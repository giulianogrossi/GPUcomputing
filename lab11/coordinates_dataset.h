#pragma once

#include "utils/matrix.h"
#include <vector>

struct CoordinatesDataset {
	size_t batch_size;
	size_t number_of_batches;
	Shape shape;
	std::vector<Matrix> batches;
	std::vector<Matrix> targets;

	// methods
	int getNumOfBatches();
	Matrix getBatch(int);
	Matrix getTarget(int);
	void fillCoordinatesDataset();
	void summary();
};
