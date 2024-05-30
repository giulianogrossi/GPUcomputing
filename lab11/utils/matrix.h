#pragma once

struct Shape {
	size_t x_dim, y_dim;

	// constructor
	Shape() : Shape(0, 0) {};
	Shape(size_t x, size_t y) : x_dim(x), y_dim(y) {}
};

struct Matrix {
	Shape shape;
	float* data_device;
	float* data_host;
	bool device_allocated;
	bool host_allocated;
	
	// constructor
	Matrix() : Matrix(0, 0) {};
	Matrix(Shape shape) : Matrix(shape.x_dim, shape.y_dim) {}
	Matrix(size_t x_dim, size_t y_dim) : shape(x_dim, y_dim), data_device(nullptr), data_host(nullptr),
													device_allocated(false), host_allocated(false) {}
	
	// methods
	void allocateMemory();
	void allocateDeviceMemory();
	void allocateHostMemory();
	void allocateMemoryIfNotAllocated(Shape);
	void copyHostToDevice();
	void copyDeviceToHost();
	void print();
	void printGPU();
};
