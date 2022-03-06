/*
 * mqdb.cpp
 */

#include "mqdb.h"

/**
 * random generate block dimensions
 */
int genRandDims(mqdb *M, uint n, uint k) {

	if (n == 0 || k == 0 || k > n) {
		printf("error: n,k must be positive and n > k!\n");
		return(-1);
	}
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
			for (int k = 0; k < M->blkSize[i]; k++)
				if (T == 'C')  	    // const fill mat entries
					M->elem[offset * n + j * n + k + offset] = c;
				else if (T == 'R') 	// random fill mat entries
					M->elem[offset * n + j * n + k + offset] = randu();
		offset += M->blkSize[i];
		M->nElems += M->blkSize[i]*M->blkSize[i];
	}
	// set description
	sprintf(M->desc, "Random mqdb:  mat. size = %d, num. blocks = %d, blk sizes: ",n,k);
}

/**
 * rand_gen_mqdb: mqdb  type returned
 *                n     square matrix size
 *                k     number of blocks
 *                seed  seed for random generator
 */
mqdb genRandMat(unsigned n, unsigned k, unsigned seed) {
	mqdb M;
	srand(seed);
	genRandDims(&M, n, k);
	M.nBlocks = k;

	// random fill mat entries
	fillBlocks(&M, n, k, 'R', 0.0);

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
	srand(seed);
	genRandDims(&M, n, k);
	M.nBlocks = k;

	// fill mat entries with a constant
	fillBlocks(&M, n, k, 'C', c);

	return M;
}

/*
 * product between mqdb matrices restricted to blocks
 */
void mqdbProd(mqdb A, mqdb B, mqdb C) {

	// TODO

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
void mqdbDisplay(mqdb M) {
	int n = 0;
	printf("%s", M.desc);
	for (int j = 0; j < M.nBlocks; j++) {
		printf("%d  ", M.blkSize[j]);
		n += M.blkSize[j];
	}
	printf("\n");
	for (int j = 0; j < n * n; j++) {
		if (M.elem[j] == 0)
			printf("------");
		else
			printf("%5.2f ", M.elem[j]);
		if ((j + 1) % n == 0)
			printf("\n");
	}
	printf("\n");
}
