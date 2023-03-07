#include <stdio.h>
#include <stdlib.h>
#include "bmpUtil.h"
#include "../../utils/common.h"

/*
 * Kernel 1D that flips the given image vertically
 * each thread only flips a single pixel (R,G,B)
 */
__global__ void VflipGPU(pel *imgDst, const pel *imgSrc, const uint w, const uint h) {
	// TODO
}

/*
 *  Kernel that flips the given image horizontally
 *  each thread only flips a single pixel (R,G,B)
 */
__global__ void HflipGPU(pel *ImgDst, pel *ImgSrc, uint width) {
	//TODO
}

/*
 *  Read a 24-bit/pixel BMP file into a 1D linear array.
 *  Allocate memory to store the 1D image and return its pointer
 */
pel *ReadBMPlin(char* fn) {
	static pel *Img;
	FILE* f = fopen(fn, "rb");
	if (f == NULL) {
		printf("\n\n%s NOT FOUND\n\n", fn);
		exit(EXIT_FAILURE);
	}

	pel HeaderInfo[54];
	size_t nByte = fread(HeaderInfo, sizeof(pel), 54, f); // read the 54-byte header
	// extract image height and width from header
	int width = *(int*) &HeaderInfo[18];
	img.width = width;
	int height = *(int*) &HeaderInfo[22];
	img.height = height;
	int s = (width * 3 + 3) & (~3);  // row is multiple of 4 pixel
	img.rowByte = s;
	//save header for re-use
	memcpy(img.headInfo, HeaderInfo, 54);
	printf("\n Input File name: %5s  (%d x %d)   File Size=%lu", fn, img.width,
			img.height, IMAGESIZE);
	// allocate memory to store the main image (1 Dimensional array)
	Img = (pel *) malloc(IMAGESIZE);
	if (Img == NULL)
		return Img;      // Cannot allocate memory
	// read the image from disk
	size_t out = fread(Img, sizeof(pel), IMAGESIZE, f);
	fclose(f);
	return Img;
}

/*
 *  Write the 1D linear-memory stored image into file
 */
void WriteBMPlin(pel *Img, char* fn) {
	FILE* f = fopen(fn, "wb");
	if (f == NULL) {
		printf("\n\nFILE CREATION ERROR: %s\n\n", fn);
		exit(1);
	}
	//write header
	fwrite(img.headInfo, sizeof(pel), 54, f);
	//write data
	fwrite(Img, sizeof(pel), IMAGESIZE, f);
	printf("\nOutput File name: %5s  (%u x %u)   File Size=%lu", fn, img.width,
			img.height, IMAGESIZE);
	fclose(f);
}

/*
 * MAIN
 */
int main(int argc, char **argv) {
	char flip = 'V';
	uint b = 256, dimGrid;
	pel *imgSrc, *imgDst;		 // Where images are stored in CPU
	pel *imgSrcGPU, *imgDstGPU;	 // Where images are stored in GPU

	if (argc > 4) {
		b = atoi(argv[4]);
		flip = argv[3][0];
	}
	else if (argc > 3) {
		flip = argv[3][0];
	}
	else if (argc < 3) {
		printf("\n\nUsage:   imflipGPU InputFilename OutputFilename [V/H] [b]\n");
		exit(EXIT_FAILURE);
	}
	if ((flip != 'V') && (flip != 'H')) {
		printf("Invalid flip option '%c'. Must be 'V','H'... \n",flip);
		exit(EXIT_FAILURE);
	}

	// Create CPU memory to store the input and output images
	imgSrc = ReadBMPlin(argv[1]); // Read the input image if memory can be allocated
	if (imgSrc == NULL) {
		printf("Cannot allocate memory for the input image...\n");
		exit(EXIT_FAILURE);
	}
	imgDst = (pel *) malloc(IMAGESIZE);
	if (imgDst == NULL) {
		free(imgSrc);
		printf("Cannot allocate memory for the input image...\n");
		exit(EXIT_FAILURE);
	}

	// Allocate GPU buffer for the input and output images
	CHECK(cudaMalloc((void**) &imgSrcGPU, IMAGESIZE));
	CHECK(cudaMalloc((void**) &imgDstGPU, IMAGESIZE));

	// Copy input vectors from host memory to GPU buffers.
	CHECK(cudaMemcpy(imgSrcGPU, imgSrc, IMAGESIZE, cudaMemcpyHostToDevice));

	// invoke kernels (define grid and block sizes)
	int rowBlock = (WIDTH + b - 1) / b;
	dimGrid = HEIGHT * rowBlock;


	double start = seconds();   // start time

	switch (flip) {
	case 'H':
		HflipGPU<<<dimGrid, b>>>(imgDstGPU, imgSrcGPU, WIDTH);
		break;
	case 'V':
		VflipGPU<<<dimGrid, b>>>(imgDstGPU, imgSrcGPU, WIDTH, HEIGHT);
		break;
	}
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	CHECK(cudaDeviceSynchronize());

	double stop = seconds();   // elapsed time

	// Copy output (results) from GPU buffer to host (CPU) memory.
	CHECK(cudaMemcpy(imgDst, imgDstGPU, IMAGESIZE, cudaMemcpyDeviceToHost));

	// Write the flipped image back to disk
	WriteBMPlin(imgDst, argv[2]);

	printf("\nKernel elapsed time %f sec \n\n", stop - start);

	// Deallocate CPU, GPU memory and destroy events.
	cudaFree(imgSrcGPU);
	cudaFree(imgDstGPU);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools spel as Parallel Nsight and Visual Profiler to show complete traces.
	cudaError_t	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		free(imgSrc);
		free(imgDst);
		exit(EXIT_FAILURE);
	}
	free(imgSrc);
	free(imgDst);
	return (EXIT_SUCCESS);
}







    


