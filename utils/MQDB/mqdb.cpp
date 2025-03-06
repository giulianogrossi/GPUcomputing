
#include "mqdb.h"

/**
 * random generate block dimensions
 */
int genRandDims(mqdb *M, uint n, uint k, int seed) {

	if (n == 0 || k == 0 || k > n) {
		printf("error: n and k must be positive and n > k!\n");
		return(-1);
	}
	srand(seed);
	M->nBlocks = k;
	// random generation of block sizes
	M->blkSize = (int *) malloc(k * sizeof(int));
	int sum = 0;
	int r;
	float mu = 2.0f * (float) n / (float) k;
	for (int i = 0; i < k - 1; i++) {
		// expected value E[block_size] = n/k
		while ((r = round(mu * randu())) > n - sum - k + i + 1);
		if (!r)
			r += 1;
		M->blkSize[i] = r;
		sum += r;
	}
	M->blkSize[k - 1] = n - sum;
	return(0);
}

/**
 * fill blocks either random or constant
 */
void fillBlocks(mqdb *M, uint n, uint k, char T, float c) {
	//mat size n*n
	M->elem = (float *) calloc(n * n, sizeof(float));
	M->nElems = 0;
	int offset = 0;
	// loop on blocks
	for (int i = 0; i < k; i++) {
		for (int j = 0; j < M->blkSize[i]; j++)
			for (int k = 0; k < M->blkSize[i]; k++) {
				if (T == 'C')  	    // const fill mat entries
					M->elem[offset * n + j * n + k + offset] = c;
				else if (T == 'R') 	// random fill mat entries
					M->elem[offset * n + j * n + k + offset] = c*randu();
				//printf("M->[%d] = %f\n", offset * n + j * n + k + offset, M->elem[offset * n + j * n + k + offset]);
	}
		offset += M->blkSize[i];
		M->nElems += M->blkSize[i]*M->blkSize[i];
	}
	// set description
	sprintf(M->desc, "Random mqdb:  mat. size = %d, num. blocks = %d",n,k);
}


/**
 * random generate block dimensions - using CUDA Unified Memory
 */
int genRandDimsUnified(mqdb *M, uint n, uint k, int seed) {

	if (n == 0 || k == 0 || k > n) {
		printf("error: n and k must be positive and n > k!\n");
		return(-1);
	}
	srand(seed);
	M->nBlocks = k;
	int sum = 0;
	int r;
	float mu = 2.0f * (float) n / (float) k;
	for (int i = 0; i < k - 1; i++) {
		// expected value E[block_size] = n/k
		while ((r = round(mu * randu())) > n - sum - k + i + 1);
		if (!r)
			r += 1;
		M->blkSize[i] = r;
		sum += r;
	}
	M->blkSize[k - 1] = n - sum;
	return(0);
}

/**
 * fill blocks either random or constant - using CUDA Unified Memory
 */
void fillBlocksUnified(mqdb *M, uint n, uint k, char T, float c) {
	//mat size n*n
	M->nElems = 0;
	int offset = 0;
	// loop on blocks
	for (int i = 0; i < k; i++) {
		for (int j = 0; j < M->blkSize[i]; j++)
			for (int k = 0; k < M->blkSize[i]; k++) {
				if (T == 'C')  	    // const fill mat entries
					M->elem[offset * n + j * n + k + offset] = c;
				else if (T == 'R') 	// random fill mat entries
					M->elem[offset * n + j * n + k + offset] = c*randu();
				//printf("M->[%d] = %f\n", offset * n + j * n + k + offset, M->elem[offset * n + j * n + k + offset]);
	}
		offset += M->blkSize[i];
		M->nElems += M->blkSize[i]*M->blkSize[i];
	}
	// set description
	sprintf(M->desc, "Random mqdb:  mat. size = %d, num. blocks = %d",n,k);
}

/**
 * rand_gen_mqdb: mqdb  type returned
 *                n     square matrix size
 *                k     number of blocks
 *                seed  seed for random generator
 */
mqdb genRandMat(unsigned n, unsigned k, unsigned seed) {
	mqdb M;
	genRandDims(&M, n, k, seed);

	// random fill mat entries
	fillBlocks(&M, n, k, 'R', 1.0);

	return M;
}

/**
 * const_mqdb: mqdb  is the type returned
 *                n     is the square matrix size
 *                k     is the number of blocks
 *                seed  is the seed for random generator
 *                c   	is the constant value assigned
 */
mqdb mqdbConst(uint n, uint k, uint seed, float c) {
	mqdb M;
	genRandDims(&M, n, k, seed);

	// fill mat entries with a constant
	fillBlocks(&M, n, k, 'C', c);

	return M;
}

/*
 * product between mqdb matrices restricted to blocks
 */
void mqdbProd(mqdb A, mqdb B, mqdb C) {
	uint n = 0;
	for (uint i = 0; i < A.nBlocks; i++)
		n += A.blkSize[i];                  // mat dim
	int k = A.nBlocks;                      // num blks
	int dl = 0;                             // blk left bound
	int dr = 0;                             // blk left bound
	for (uint i = 0; i < k; i++) {           // loop on blks
		dr += A.blkSize[i];                 // blk right bound
		for (uint r = dl; r < dr; r++) {     // scan block rows
			for (uint c = dl; c < dr; c++) { // scan block cols
				float s = 0;
				for (uint l = dl; l < dr; l++)
					s += A.elem[r*n + l] * B.elem[c + l * n];
				C.elem[r*n + c] = s;
			}
		}
		dl = dr;
	}
}

/*
 * standard (naive) matrix product on host
 */
void matProd(mqdb A, mqdb B, mqdb C) {
	int n = 0;
	for (uint i = 0; i < A.nBlocks; i++)
		n += A.blkSize[i];

	for (uint r = 0; r < n; r++)
		for (uint c = 0; c < n; c++) {
			double sum = 0;
			for (uint l = 0; l < n; l++){
				double a = A.elem[r * n + l];
				double b = B.elem[l * n + c];
				sum += a*b;
			}
			C.elem[r * n + c] = (float)sum;
		}
}

/*
 * elementwise comparison between two mqdb
 */
void checkResult(mqdb A, mqdb B) {
	double epsilon = 1.0E-8;
	bool match = 1;
	int n = 0;
	for (int i = 0; i < A.nBlocks; i++)
		n += A.blkSize[i];
	for (int i = 0; i < n * n; i++) {
		if (abs(A.elem[i] - B.elem[i]) > epsilon) {
			match = 0;
			printf("   * Arrays do not match!\n");
			printf("     gpu: %2.2f,  host: %2.2f at current %d\n", A.elem[i],
					B.elem[i], i);
			break;
		}
	}
	if (match)
		printf("   Arrays match\n\n");
}
/*
 * print mqdb
 */
void mqdbDisplay(mqdb *M) {
	int k = M->nBlocks;
	int n = 0;
	printf("%s\n", M->desc);
	printf("Block sizes [%d]: ", k);
	for (int j = 0; j < k; j++) {
		n += M->blkSize[j];
		printf("%d  ", M->blkSize[j]);
	}

	printf("\nElements: \n");
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (M->elem[i*n + j] == 0)
				printf("------");
			else
				printf("%5.2f ", M->elem[i*n + j]);
		}
		printf("\n");
	}
	printf("\n");
}