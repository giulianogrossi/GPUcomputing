#include <stdio.h>
#include <stdlib.h>

int mycudaMalloc(void **devPtr, size_t size) {
    printf("Addresses p_d (inside1)  = %p\n", *devPtr);
    *devPtr = malloc(size);
    printf("Addresses p_d (inside2)  = %p\n", *devPtr);
    if (*devPtr == NULL) {
        fprintf(stderr, "Error: malloc failed\n");
        return -1;
    } else {
        printf("malloc success\n");
         return 0;
    }
}
/*
 Provide a Bus Error (also known as SIGBUS and is usually signal 10): 
 occur when a process is trying to access memory that the CPU cannot 
 physically address. In other words the memory tried to access by the 
 program is not a valid memory address. It caused due to alignment 
 issues with the CPU (eg. trying to read a long from an address which 
 isn’t a multiple of 4). It is a common error in Unix/Linux systems. 
*/
int mycudaMalloc_fake(void *devPtr, size_t size) {
    devPtr = malloc(size);
    printf("Addresses p_d (inside)     = %p\n", devPtr);
    if (devPtr == NULL) {
        fprintf(stderr, "Error: malloc failed\n");
        return -1;
    } else {
        printf("malloc success\n");
        return 0;
    }
}

int main(void) {
    printf("Test di mycudaMalloc...\n");

    int *p_d;
    printf("Address p_d (pre alloc)  = %p\n", p_d);
    mycudaMalloc((void **)&p_d, sizeof(int) * 10);
    printf("Address p_d (post alloc) = %p\n", p_d);

    printf("\n\nTest di mycudaMalloc_fake...\n");        
    int *q_d;
    printf("Address p_d (pre alloc)  = %p\n", q_d);
    mycudaMalloc((void *)q_d, sizeof(int) * 10);
    printf("Address p_d (post alloc) = %p\n", q_d);

    return 0;
}