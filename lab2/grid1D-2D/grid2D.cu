#include <stdio.h>

/*
 * Show DIMs & IDs for grid, block and thread
 */
__global__ void checkIndex(void) {
	if ((threadIdx.x + threadIdx.y) && ((threadIdx.x + threadIdx.y) % 5 == 0)) {
		printf("threadIdx:(%d, %d, %d) blockIdx:(%d, %d, %d) "
				"blockDim:(%d, %d, %d) gridDim:(%d, %d, %d)\n", threadIdx.x,
				threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z,
				blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y,
				gridDim.z);
	}
}

int main(int argc, char **argv) {

	// grid and block structure
	dim3 block(7,6);
	dim3 grid(2,2);

	// check for host
	printf("CHECK for host:\n");
	printf("grid.x = %d\t grid.y = %d\t grid.z = %d\n", grid.x, grid.y, grid.z);
	printf("block.x = %d\t block.y = %d\t block.z %d\n", block.x, block.y, block.z);

	// check for device
	printf("CHECK for device:\n");
	checkIndex<<<grid, block>>>();

	// reset device
	cudaDeviceReset();
	return (0);
}
