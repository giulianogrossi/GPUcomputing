#include <stdio.h>
#include <stdlib.h>
#include "bmpUtil.h"
#include "common.h"

/*
 * Kernel 1D that flips the given image vertically
 * each thread only flips a single pixel (R,G,B)
 */
__global__ void Vflip(pel *imgDst, const pel *imgSrc, const uint w, const uint h) {
	// ** pixel granularity **
	uint i = blockIdx.x;               // block ID
	uint j = threadIdx.x;              // thread ID
	uint b = blockDim.x;               // block dim
	uint x = b * i + j;                // 1D pixel linear index
	uint m = (w + b - 1) / b;          // num of blocks in a row
	uint r = i / m;                    // row of the source pixel
	uint c = x - r * w;                // col of the source pixel

	if (c >= w)                        // col out of range
		return;

	//  ** byte granularity **
	uint s = (w * 3 + 3) & (~3);       // num bytes x row (mult. 4)
	uint r1 = h - 1 - r;               // dest. row (mirror)
	uint p = s * r + 3*c;              // src byte position of the pixel
	uint q = s * r1 + 3*c;             // dst byte position of the pixel
	// swap pixels RGB
	imgDst[q] = imgSrc[p];             // R
	imgDst[q + 1] = imgSrc[p + 1];     // G
	imgDst[q + 2] = imgSrc[p + 2];     // B
}

/*
 *  Kernel that flips the given image horizontally
 *  each thread only flips a single pixel (R,G,B)
 */
__global__ void Hflip(pel *ImgDst, pel *ImgSrc, uint width) {
	uint dimBlock = blockDim.x;
	uint MYbid = blockIdx.x;
	uint MYtid = threadIdx.x;
	uint MYgtid = dimBlock * MYbid + MYtid;

	uint BlkPerRow = (width + dimBlock - 1) / dimBlock;  // ceil
	uint RowBytes = (width * 3 + 3) & (~3);
	uint MYrow = MYbid / BlkPerRow;
	uint MYcol = MYgtid - MYrow * BlkPerRow * dimBlock;
	if (MYcol >= width)
		return;			// col out of range
	uint MYmirrorcol = width - 1 - MYcol;
	uint MYoffset = MYrow * RowBytes;
	uint MYsrcIndex = MYoffset + 3 * MYcol;
	uint MYdstIndex = MYoffset + 3 * MYmirrorcol;

	// swap pixels RGB   @MYcol , @MYmirrorcol
	ImgDst[MYdstIndex] = ImgSrc[MYsrcIndex];
	ImgDst[MYdstIndex + 1] = ImgSrc[MYsrcIndex + 1];
	ImgDst[MYdstIndex + 2] = ImgSrc[MYsrcIndex + 2];
}

/*
 * MAIN
 */
int main(int argc, char **argv) {
	char flip = 'V';
	uint dimBlock = 256, dimGrid;
	pel *imgSrc, *imgDst;		 // Where images are stored in CPU
	pel *imgSrcGPU, *imgDstGPU;	 // Where images are stored in GPU

	if (argc > 4) {
		dimBlock = atoi(argv[4]);
		flip = argv[3][0];
	}
	else if (argc > 3) {
		flip = argv[3][0];
	}
	else if (argc < 3) {
		printf("\n\nUsage:   imflipG InputFilename OutputFilename [V/H] [dimBlock]");
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
	int rowBlock = (WIDTH + dimBlock - 1) / dimBlock;
	dimGrid = HEIGHT * rowBlock;

	double start = seconds();   // start time

	switch (flip) {
	case 'H':
		Hflip<<<dimGrid, dimBlock>>>(imgDstGPU, imgSrcGPU, WIDTH);
		break;
	case 'V':
		Vflip<<<dimGrid, dimBlock>>>(imgDstGPU, imgSrcGPU, WIDTH, HEIGHT);
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
	int RowBytes = (width * 3 + 3) & (~3);  // row is multiple of 4 pixel
	img.rowByte = RowBytes;
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

