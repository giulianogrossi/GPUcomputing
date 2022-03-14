#include <stdio.h>

/*
 * Mostra DIMs e IDs di grid, block e thread
 */
__global__ void checkIndex(void) {
	printf("threadIdx:(%d, %d, %d) blockIdx:(%d, %d, %d) "
			"blockDim:(%d, %d, %d) gridDim:(%d, %d, %d)\n",
			threadIdx.x, threadIdx.y, threadIdx.z,
			blockIdx.x, blockIdx.y, blockIdx.z,
			blockDim.x, blockDim.y, blockDim.z,
			gridDim.x,gridDim.y,gridDim.z);
}

int main(int argc, char **argv) {

	// definisce grid e struttura dei blocchi
	dim3 block(4);
	dim3 grid(3);

	// controlla dim. dal lato host
	printf("CHECK lato host:\n");
	printf("grid.x = %d\t grid.y = %d\t grid.z = %d\n", grid.x, grid.y, grid.z);
	printf("block.x = %d\t block.y = %d\t block.z %d\n", block.x, block.y, block.z);

	// controlla dim. dal lato device
	printf("CHECK lato device:\n");
	checkIndex<<<grid, block>>>();

	// reset device
	cudaDeviceReset();
	return(0);
}
