
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "../../utils/common.h"

#define THREADS 16
#define BLOCKS 1


/*
 * Indici k e j in input hanno significato:
 *     k = 2,4,8,...,2^s=N
 *     j = 2^(k-1), 2^(k-2),...,1 (parte dalla metà di k e continua a dimezzare)
 * Gli operatori sui bit ^ (XOR) e & (AND) vengono usati per filtrare i thread:
 *     ixj = i ^ j  aggiunge o toglie a i una potenza di 2, cioé ixj = i +- j (j = 2^a)
 *     i & k == 0   vero sse i <= k (sort ascendente) altrimenti sort discendente
 * L'operazione ixj > i significa aggiorna solo quando l'indice ixj fa un salto in
 * avanti di j = 2^a
 * Funzionamento:
 */
__global__ void bitonic_sort_step(int *a, int j, int k) {
	unsigned int i, ixj;                       // Sorting partners i and ixj
	i = threadIdx.x + blockDim.x * blockIdx.x;
	ixj = i ^ j;   // XOR: aggiunge o toglie a i una potenza di 2, j = 2^a

	if (i == 0)
		printf("ROUND: k = %d, j = %d\n", k, j);

	if ((ixj) > i) {    // entra solo quando fa un salto di j = 2^a

//		 Sort ascending
		if ((i & k) == 0) {
			printf("  UP  (ixj = %d\t    i = %d\t k = %d)   a[ixj] = %d - a[i] = %d\n", ixj, i, k, a[ixj],a[i]);
			if (a[i] > a[ixj]) {
				int temp = a[i];
				a[i] = a[ixj];
				a[ixj] = temp;
			}
		}

		// Sort descending
		if ((i & k) != 0) {
			printf("  DOWN  (ixj = %d\t    i = %d\t k = %d)   a[ixj] = %d - a[i] = %d\n", ixj, i, k, a[ixj],a[i]);
			if (a[i] < a[ixj]) {
				int temp = a[i];
				a[i] = a[ixj];
				a[ixj] = temp;
			}
		}
	}
}

/*The parameter dir indicates the sorting direction, ASCENDING
 or DESCENDING; if (a[i] > a[j]) agrees with the direction,
 then a[i] and a[j] are interchanged.*/
void compAndSwap(int a[], int i, int j, int dir) {
	if (dir == (a[i] > a[j])) {
		int tmp = a[i];
		a[i] = a[j];
		a[j] = tmp;
	}
}

/*It recursively sorts a bitonic sequence in ascending order,
 if dir = 1, and in descending order otherwise (means dir=0).
 The sequence to be sorted starts at index position low,
 the parameter cnt is the number of elements to be sorted.*/
void bitonicMerge(int a[], int low, int cnt, int dir) {
	if (cnt > 1) {
		int k = cnt / 2;
		for (int i = low; i < low + k; i++)
			compAndSwap(a, i, i + k, dir);
		bitonicMerge(a, low, k, dir);
		bitonicMerge(a, low + k, k, dir);
	}
}

/* This function first produces a bitonic sequence by recursively
 sorting its two halves in opposite sorting orders, and then
 calls bitonicMerge to make them in the same order */
void bitonicSort(int a[], int low, int cnt, int dir) {
	if (cnt > 1) {
		int k = cnt / 2;

		// sort in ascending order since dir here is 1
		bitonicSort(a, low, k, 1);

		// sort in descending order since dir here is 0
		bitonicSort(a, low + k, k, 0);

		// Will merge wole sequence in ascending order
		// since dir=1.
		bitonicMerge(a, low, cnt, dir);
	}
}

/*
 * test bitonic sort on CPU and GPU
 */
int main(void) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int N = THREADS*BLOCKS;
	// check
	if (!(N && !(N & (N - 1)))) {
		printf("ERROR: N must be power of 2 (N = %d)\n", N);
		exit(1);
	}
	size_t nBytes = N * sizeof(int);
	int *a = (int*) malloc(nBytes);
	int *b = (int*) malloc(nBytes);

	// fill data
	for (int i = 0; i < N; ++i) {
		a[i] =  i%5; //rand() % 100; // / (float) RAND_MAX;
		b[i] = a[i];
	}

	// bitonic CPU
	double cpu_time = seconds();
	bitonicSort(b, 0, N, 1);   // 1 means sort in ascending order
	printf("CPU elapsed time: %.5f (sec)\n", seconds()-cpu_time);

	// device mem copy
	int *d_a;
	CHECK(cudaMalloc((void**) &d_a, nBytes));
	CHECK(cudaMemcpy(d_a, a, nBytes, cudaMemcpyHostToDevice));

	// num of threads
	dim3 blocks(BLOCKS, 1);   // Number of blocks
	dim3 threads(THREADS, 1); // Number of threads

	// start computation
	cudaEventRecord(start);
	int j, k;
	// external loop on comparators of size k
	for (k = 2; k <= N; k <<= 1) {
		// internal loop for comparator internal stages
		for (j = k >> 1; j > 0; j = j >> 1)
			bitonic_sort_step<<<blocks, threads>>>(d_a, j, k);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("GPU elapsed time: %.5f (sec)\n", milliseconds / 1000);

	// recover data
	cudaMemcpy(a, d_a, nBytes, cudaMemcpyDeviceToHost);

	// print & check
	if (N < 100) {
		printf("GPU:\n");
		for (int i = 0; i < N; ++i)
			printf("%d\n", a[i]);
		printf("CPU:\n");
		for (int i = 0; i < N; ++i)
			printf("%d\n", b[i]);
	}
	else {
		for (int i = 0; i < N; ++i) {
			if (a[i] != b[i]) {
				printf("ERROR a[%d] != b[%d]  (a[i] = %d  -  b[i] = %d\n", i,i, a[i],b[i]);
				break;
			}
		}
	}

	cudaFree(d_a);
	exit(0);
}



