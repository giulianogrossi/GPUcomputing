#include <sys/time.h>
#include "mqdb.h"

/*
 * main function
 */
int main(void) {
	uint n = 2*1024;      // matrix size
   uint k = 10;          // num of blocks
	mqdb A, B, C, C1;     // mqdb host matrices

	// # fill in #
	A = mqdbConst(n, k, 10, 1);
	B = mqdbConst(n, k, 10, 1);
	C = mqdbConst(n, k, 10, 1);
	C1 = mqdbConst(n, k, 10, 1);

	ulong nBytes = n * n * sizeof(float);
	ulong kBytes = k * sizeof(uint);
	printf("Memory size required = %.1f (MB)\n",(float)nBytes/(1024.0*1024.0));

	printf("CPU mat product...\n");
	double start = seconds();
   matProd(A, B, C);
	double CPUTime = seconds() - start;
	printf("CPU elapsed time: %.5f (sec)\n\n", CPUTime);

   printf("CPU MQDB product...\n");
	start = seconds();
   mqdbProd(A, B, C1);
	CPUTime = seconds() - start;
	printf("CPU elapsed time: %.5f (sec)\n\n", CPUTime);

	// check result
	checkResult(C, C1);
 
	return 0;
}
