#include <stdlib.h>
#include <stdio.h>

/**
 *  Merge the two half into a sorted data
 */
void merge(int arr[], int l, int m, int r) {
	//	printf("merge: l = %d, m = %d, r = %d\n",l,m,r);
	int i, j, k;
	int n1 = m - l + 1;
	int n2 = r - m;

	// Create temp arrays
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
