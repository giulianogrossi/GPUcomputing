
/*
* Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/
#include <iostream>
#include <cstdio>
#include "../../utils/common.h"

#define MAX_DEPTH       16
#define INSERTION_SORT  32

////////////////////////////////////////////////////////////////////////////////
// Selection sort used when depth gets too big or the number of elements drops
// below a threshold.
////////////////////////////////////////////////////////////////////////////////
__device__ void selection_sort(unsigned int *data, int left, int right) {
    for (int i = left ; i <= right ; ++i) {
        unsigned min_val = data[i];
        int min_idx = i;

        // Find the smallest value in the range [left, right].
        for (int j = i+1 ; j <= right ; ++j) {
            unsigned val_j = data[j];

            if (val_j < min_val) {
                min_idx = j;
                min_val = val_j;
            }
        }

        // Swap the values.
        if (i != min_idx) {
            data[min_idx] = data[i];
            data[i] = min_val;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Very basic quicksort algorithm, recursively launching the next level.
////////////////////////////////////////////////////////////////////////////////
__global__ void cdp_simple_quicksort(unsigned int *data, int left, int right, int depth) {
    // If we're too deep or there are few elements left, we use an insertion sort...
    if (depth >= MAX_DEPTH || right-left <= INSERTION_SORT) {
        selection_sort(data, left, right);
        return;
    }

    unsigned int *lptr = data+left;
    unsigned int *rptr = data+right;
    unsigned int  pivot = data[(left+right)/2];

    // Do the partitioning.
    while (lptr <= rptr) {
        // Find the next left- and right-hand values to swap
        unsigned int lval = *lptr;
        unsigned int rval = *rptr;

        // Move the left pointer as long as the pointed element is smaller than the pivot.
        while (lval < pivot) {
            lptr++;
            lval = *lptr;
        }

        // Move the right pointer as long as the pointed element is larger than the pivot.
        while (rval > pivot) {
            rptr--;
            rval = *rptr;
        }

        // If the swap points are valid, do the swap!
        if (lptr <= rptr) {
            *lptr++ = rval;
            *rptr-- = lval;
        }
    }

    // Now the recursive part
    int nright = rptr - data;
    int nleft  = lptr - data;

    // Launch a new block to sort the left part.
    if (left < (rptr-data)) {
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        cdp_simple_quicksort<<< 1, 1, 0, s >>>(data, left, nright, depth+1);
        cudaStreamDestroy(s);
    }

    // Launch a new block to sort the right part.
    if ((lptr-data) < right) {
        cudaStream_t s1;
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        cdp_simple_quicksort<<< 1, 1, 0, s1 >>>(data, nleft, right, depth+1);
        cudaStreamDestroy(s1);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Call the quicksort kernel from the host.
////////////////////////////////////////////////////////////////////////////////
void run_qsort(unsigned int *data, unsigned int nitems) {
    // Prepare CDP for the max depth 'MAX_DEPTH'.
    CHECK(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH));

    // Launch on device
    int left = 0;
    int right = nitems-1;
    std::cout << "Launching kernel on the GPU" << std::endl;
    cdp_simple_quicksort<<< 1, 1 >>>(data, left, right, 0);
    CHECK(cudaDeviceSynchronize());
}

////////////////////////////////////////////////////////////////////////////////
// Initialize data on the host.
////////////////////////////////////////////////////////////////////////////////
void initialize_data(unsigned int *dst, unsigned int nitems) {
    // Fixed seed for illustration
    srand(2047);

    // Fill dst with random values
    for (unsigned i = 0 ; i < nitems ; i++)
        dst[i] = rand() % nitems ;
}

////////////////////////////////////////////////////////////////////////////////
// Verify the results.
////////////////////////////////////////////////////////////////////////////////
void check_results(int n, unsigned int *results_d) {
    unsigned int *results_h = new unsigned[n];
    CHECK(cudaMemcpy(results_h, results_d, n*sizeof(unsigned), cudaMemcpyDeviceToHost));

    for (int i = 1 ; i < n ; ++i)
        if (results_h[i-1] > results_h[i]) {
            std::cout << "Invalid item[" << i-1 << "]: " << results_h[i-1] << " greater than " << results_h[i] << std::endl;
            exit(EXIT_FAILURE);
        }

    std::cout << "OK" << std::endl;
    delete[] results_h;
}

////////////////////////////////////////////////////////////////////////////////
// Main entry point.
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    int num_items = 1024*1024;
    bool verbose = false;

    // Get device properties
    int device = 0;
    
    cudaSetDevice(device);

    // Create input data
    unsigned int *h_data = 0;
    unsigned int *d_data = 0;

    // Allocate CPU memory and initialize data.
    std::cout << "Initializing data:" << std::endl;
    h_data =(unsigned int *)malloc(num_items*sizeof(unsigned int));
    initialize_data(h_data, num_items);

    if (verbose) {
        for (int i=0 ; i<num_items ; i++)
            std::cout << "Data [" << i << "]: " << h_data[i] << std::endl;
    }

    // Allocate GPU memory.
    CHECK(cudaMalloc((void **)&d_data, num_items * sizeof(unsigned int)));
    CHECK(cudaMemcpy(d_data, h_data, num_items * sizeof(unsigned int), cudaMemcpyHostToDevice));

    // Execute
    std::cout << "Running quicksort on " << num_items << " elements" << std::endl;
    run_qsort(d_data, num_items);

    // Check result
    std::cout << "Validating results: ";
    check_results(num_items, d_data);

    free(h_data);
    CHECK(cudaFree(d_data));

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
    exit(EXIT_SUCCESS);
}

