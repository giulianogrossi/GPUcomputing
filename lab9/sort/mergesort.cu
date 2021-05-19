
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "../../utils/common.h"

#define ARRAY_SIZE 64

void check_up_sorting(int *, unsigned);
void random_array(int *, unsigned, int);
void printArray(int *, int, int);
void mergeSort(int *, int, int);
void merge(int *, int, int, int);
void arrayCopy(int *, const int *, const int);


/**
 * Kernel: mergeSort with seq. merge
 */
__global__ void cudaMergeSort(int *array, int *sorted, int n, int chunk) {

	int start = chunk * (threadIdx.x + blockIdx.x * blockDim.x);
	if (start > n - chunk)
		return;

	int mid = start + chunk / 2;
	int end = start + chunk;
	int i = start, j = mid, k = start;

	//cudaMerge(array, sorted, start, mid, end);
	while (i < mid && j < end) 
		if (array[i] <= array[j]) 
			sorted[k++] = array[i++];
		else 
			sorted[k++] = array[j++];

	// Copy the remaining elements array[i] if there are any
	while (i < mid)
		sorted[k++] = array[i++];

	// Copy the remaining elements of array[j] if there are any
	while (j < end)
		sorted[k++] = array[j++];
}

/**
 * A iterative binary search function. It returns the location p of
 * the first element in r-length arr[0..r-1] greater than x
 */
__device__ int binarySearch(int arr[], int x, int k, bool UP) {
	int l = 0, r = k;

	while (l < r) {
		int m = (l+r)/2;
		if (UP) {     // for upper chunk B
			if (arr[m] <= x) l = m + 1;
			else r = m;
		}
		else {   // for lower chunk A
			if (arr[m] < x) l = m + 1;
			else r = m;
		}
	}

	return l;
}

/**
 * Kernel: mergeSort with many threads. Each thread deals with 2 elements:
 *  A[i] in first chunk and the corresponding B[i] in the second chunk
 */
__global__ void cudaMergeSortMulti(int *array, int *sorted, int n, int k) {
	// k = 1,2,4,8,16,..., 2^m chunk dims
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int j = tid % k;
	int l = (tid - j)*2;  // first element of the fisrt chunk
	int i = l + j;        // A[i] first chunk   [][][*][] and  B[i+k] [][][*][]

	if (k == 1) {
		l = 2*tid;
		i = l;
	}

	// find the relative position of x within B[*]
	int x = array[i];
	int p = binarySearch(array+l+k, x, k, 1);
	sorted[i+p] = x;

	// find the relative position of y within A[*]
	int y = array[i+k];
	p = binarySearch(array+l, y, k, 0);
	sorted[i+p] = y;
}

/*
 * MAIN
 */
int main(int argc, char** argv) {

	// Create the vector with the specified size and situation
	int *orig, *array, *sorted;
	int N = 4*1024*1024;         // must be a power of 2
	int BLOCK_SIZE = 32;


	printf("*** Sorting array size N = %d\n",N);

	// events to measure time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// managed memory
	CHECK(cudaMallocManaged((void **)&array, N * sizeof(int)));
	CHECK(cudaMallocManaged((void **)&sorted, N * sizeof(int)));

	// random instance
	orig = (int *) malloc(N * sizeof(int));
	random_array(orig, N, 1);
	arrayCopy(array, orig, N);
	// printArray(array,N,16);

	/*****************************************************
	 *                      CPU                          *
	 *****************************************************/
	
  printf("*** CPU processing...\n");
	double startTm = seconds();
	mergeSort(array, 0, N);
	double CPUtime = seconds() - startTm;
	printf("   CPU elapsed time: %.5f (sec)\n", CPUtime);
	check_up_sorting(array, N);

	/*****************************************************
	 *              ONE THREAD x chunk                   *
	 *****************************************************/
	
  printf("\n*** GPU ONE THREAD x chunk processing...\n");
	arrayCopy(sorted, orig, N); // start from step 2
	bool array2sorted = false;
	CHECK(cudaEventRecord(start));
	for (int chunk = 2; chunk <= N; chunk *= 2) {
		int nThreads = N / chunk;
		dim3 block(min(nThreads, BLOCK_SIZE));
		dim3 grid((nThreads + block.x - 1) / block.x);

		if (array2sorted)
			cudaMergeSort<<<grid, block>>>(array, sorted, N, chunk);
		else
			cudaMergeSort<<<grid, block>>>(sorted, array, N, chunk);
		array2sorted = !array2sorted;
	}
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaEventRecord(stop));
	CHECK(cudaEventSynchronize(stop));
	float milliseconds;
	cudaEventElapsedTime(&milliseconds, start, stop);
	float GPUtime = milliseconds / 1000.0;
	printf("   elapsed time:   %.5f (sec)\n", GPUtime);
	printf("   speedup vs CPU: %.2f\n", CPUtime / GPUtime);

	check_up_sorting(sorted, N);

	/*****************************************************
	 *              MULTI THREAD x chunk                 *
	 *****************************************************/
	printf("\n*** GPU MULTI THREAD x chunk processing...\n");
	arrayCopy(array, orig, N);
	array2sorted = false;

	// grid set up
	int nThreads = N/2;
	dim3 block(min(nThreads, BLOCK_SIZE));
	dim3 grid((nThreads + block.x - 1) / block.x);
	CHECK(cudaEventRecord(start));
	for (int chunk = 1; chunk <= N/2; chunk *= 2) {
		array2sorted = !array2sorted;
		if (array2sorted)
			cudaMergeSortMulti<<<grid, block>>>(array, sorted, N, chunk);
		else
			cudaMergeSortMulti<<<grid, block>>>(sorted, array, N, chunk);
	}
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaEventRecord(stop));
	CHECK(cudaEventSynchronize(stop));
	cudaEventElapsedTime(&milliseconds, start, stop);
	float GPUtime1 = milliseconds / 1000.0;
	printf("   elapsed time:        %.5f (sec)\n", GPUtime1);
	printf("   speedup vs CPU:      %.2f\n", CPUtime / GPUtime1);
	printf("   speedup vs GPU mono: %.2f\n", GPUtime / GPUtime1);
	if (!array2sorted) {
		int *swap = sorted;
		sorted = array;
		array = swap;
	}
	check_up_sorting(sorted, N);

//	printArray(array,N,32);
//	printArray(sorted,N,64);

	return 0;

}

/**
 *  Merge the two half into a sorted data
 */
void merge(int arr[], int l, int m, int r) {
	//	printf("merge: l = %d, m = %d, r = %d\n",l,m,r);
	int i, j, k;
	int n1 = m - l + 1;
	int n2 = r - m;

	// Create temp arrays
	//	int L[n1], R[n2];
	int *L = new int[n1];
	int *R = new int[n2];

	// Copy data to temp arrays L[] and R[]
	for (i = 0; i < n1; i++)
		L[i] = arr[l + i];
	for (j = 0; j < n2; j++)
		R[j] = arr[m + 1 + j];

	// Merge the temp arrays back into arr[l..r]

	i = 0; // Initial index of first subarray
	j = 0; // Initial index of second subarray
	k = l; // Initial index of merged subarray

	while (i < n1 && j < n2) {
		if (L[i] <= R[j]) {
			arr[k] = L[i];
			i++;
		} else {
			arr[k] = R[j];
			j++;
		}
		k++;
	}

	// Copy the remaining elements of L[], if there are any
	while (i < n1) {
		arr[k] = L[i];
		i++;
		k++;
	}

	// Copy the remaining elements of R[], if there are any
	while (j < n2) {
		arr[k] = R[j];
		j++;
		k++;
	}
}

// l is for left index and r is right index of the
// sub-array of arr to be sorted
void mergeSort(int arr[], int l, int r) {
	//	printf("mergeSort: l = %d, r = %d\n",l,r);
	if (l < r) {
		int m = (l + r) / 2;

		// Sort first and second halves
		mergeSort(arr, l, m);
		mergeSort(arr, m + 1, r);

		// merge in backtracking step
		merge(arr, l, m, r);
	}
}

/**
 * Function that fills an array with random integers
 * @param int* array Reference to the array that will be filled
 * @param int  size  Number of elements
 */
void random_array(int *array, unsigned size, int seed) {
	srand(seed);
	int i;
	for (i = 0; i < size; i++) {
		array[i] = rand() % size;
	}
}

/**
 * Function that checks whether the sorting is correct
 * @param int* array Reference to the array
 * @param int  size  Number of elements
 */
void check_up_sorting(int array[], unsigned size) {
	bool flag = true;
	for (int i = 0; i < size - 1; i++)
		if (array[i] > array[i + 1]) {
			printf("Sorting error! array[%d]=%d array[%d]=%d\n", i, array[i], i + 1,
					array[i + 1]);
			flag = false;
			break;
		}
	if (flag)
		printf("   Sorting OK!\n");
}

/*
 * Function to print an array
 */
void printArray(int arr[], int size, int k) {
	for (int i = 0; i < size; i++) {
		if (i>0 && k > 0 && i%k==0)
			printf("\n");
		printf("%d ", arr[i]);
	}
	printf("\n\n");
}

/*
 * Function to print an array
 */
void arrayCopy(int dst[], const int src[], const int size) {
	for (int i = 0; i < size; i++)
		dst[i] = src[i];
}

