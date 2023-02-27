#include <stdio.h>
#include <iostream>

using namespace std;

__global__ void helloFromGPU (void) {
    int tID = threadIdx.x;
    printf("Hello World from GPU (I'am thread = %d)!\n", tID);
}

int main(void) {
    // hello from GPU 
    cout << "Hello World from CPU!" << endl;
    cudaSetDevice(0);
    helloFromGPU <<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}