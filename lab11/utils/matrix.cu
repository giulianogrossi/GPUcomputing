#include "matrix.h"
#include "cuda_API_check.h"
#include <stdio.h>
#include <iostream>

void Matrix::allocateMemory() {
	allocateHostMemory();
	allocateDeviceMemory();
}

void Matrix::allocateDeviceMemory() {
	if (!device_allocated) {
		CHECK(cudaMalloc((float**)&data_device, shape.x_dim * shape.y_dim * sizeof(float)));
		device_allocated = true;
	}
}

void Matrix::allocateHostMemory() {
	if (!host_allocated) {
		data_host = new float[shape.x_dim * shape.y_dim];
		host_allocated = true;
	}
}

void Matrix::print() {
	printf("Mat values:\n");
  	for (int i = 0; i < shape.y_dim; i++) {
  		for (int j = 0; j < shape.x_dim; j++)
    		printf("%8.4f", data_host[i * shape.x_dim + j]);
		printf("\n");
	}
}

void Matrix::printGPU() {
	Matrix A(shape);
	A.allocateMemory();

	CHECK(cudaMemcpy(A.data_host, data_device, shape.x_dim * shape.y_dim * sizeof(float), cudaMemcpyDeviceToHost));
	
	printf("Mat values (GPU):\n");
  	for (int i = 0; i < shape.y_dim; i++) {
  		for (int j = 0; j < shape.x_dim; j++)
    		printf("%8.4f", A.data_host[i * shape.x_dim + j]);
		printf("\n");
	}
}

void Matrix::allocateMemoryIfNotAllocated(Shape shape) {
	if (!device_allocated && !host_allocated) {
		this->shape = shape;
		// std::cout << "shape:" << shape.y_dim << std::endl;
		allocateMemory();
	}
}

void Matrix::copyHostToDevice() {
	if (device_allocated && host_allocated) {
		CHECK(cudaMemcpy(data_device, data_host, shape.x_dim * shape.y_dim * sizeof(float), cudaMemcpyHostToDevice));
	}
	else {
		std::cout << "WARN: Cannot copy host data to not allocated memory on device." << std::endl;
	}
}

void Matrix::copyDeviceToHost() {
	if (device_allocated && host_allocated) {
		CHECK(cudaMemcpy(data_host, data_device, shape.x_dim * shape.y_dim * sizeof(float), cudaMemcpyDeviceToHost));
	} 
	else {
		std::cout << "WARN: Cannot copy device data to not allocated memory on host." << std::endl;
	}
}
