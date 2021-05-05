
#include <stdio.h>
#include <stdlib.h>
#include "cublas_v2.h"
#include "../../utils/common.h"

#define IDX2R(r,c,D) ( r * D + c) 
#define IDX2C(r,c,D) ( c * D + r )

#define BLOCK_SIZE 4
#define M          (1<<14)
#define N          (1<<14)
#define P          (1<<14)


void generate_random_vector(int, float**);
void generate_random_dense_matrix_Row_Maj(int, int, float**);
void generate_random_dense_matrix_Col_Maj(int, int, float**);
void plot_mat_Row_Maj(int, int, float*, char);
void plot_mat_Col_Maj(int, int, float*, char);
__global__ void matProdSMEMstatic(float*, float*, float*, int, int, int);

/*
 * confronto tra prodotti matriciali con kernel standard e cuBLAS
 */
int main(int argc, char **argv) {

	int n = N, m = M, p = P;
	float *A, *d_A;  // matrix M x N  (row M, col N)
	float *B, *d_B;  // matrix N x P  (row N, col P)
	float *C, *d_C;  // matrix M x P, C = A*B
	float *x, *d_x;  // vector N x 1 
	float *y, *d_y;  // vector N x 1, y = A*x
	float beta = 0.0f;
	float alpha = 1.0f;
	cublasHandle_t handle;
	device_name();

	// events to measure time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Generate inputs
	srand(10);
	generate_random_dense_matrix_Col_Maj(m, n, &A);
	generate_random_dense_matrix_Col_Maj(n, p, &B);
	generate_random_vector(n, &x);
	generate_random_vector(n, &y);


	C = (float *) malloc(m * p * sizeof(float));

	// Allocate device memory
	CHECK(cudaMalloc((void **)&d_A, m * n * sizeof(float)));
	CHECK(cudaMalloc((void **)&d_B, n * p * sizeof(float)));
	CHECK(cudaMalloc((void **)&d_C, m * p * sizeof(float)));
	CHECK(cudaMalloc((void **)&d_x, n * sizeof(float)));
	CHECK(cudaMalloc((void **)&d_y, m * sizeof(float)));

	// Create the cuBLAS handle
	CHECK_CUBLAS(cublasCreate(&handle));
	int version;
	CHECK_CUBLAS(cublasGetVersion(handle, &version));
	printf("Using CUBLAS Version: %d\n", version);
	
	// Transfer inputs to the device, column-major order
	CHECK_CUBLAS(cublasSetMatrix(m, n, sizeof(float), A, m, d_A, m));
	CHECK_CUBLAS(cublasSetMatrix(n, p, sizeof(float), B, n, d_B, n));
	CHECK_CUBLAS(cublasSetMatrix(m, p, sizeof(float), C, m, d_C, m));
	CHECK_CUBLAS(cublasSetVector(n, sizeof(float), x, 1, d_x, 1));
	CHECK_CUBLAS(cublasSetVector(m, sizeof(float), y, 1, d_y, 1));

	/***************************************************
	 *      Moltiplicazione matrix-vector CUBLAS       *
	 ***************************************************/
	
  printf("\n**  Matrix-vector product...\n");
  printf("    y(%d x 1) = A(%d x %d) * x(%d x 1)\n",n,m,n,n);

	cudaEventRecord(start);
	CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_N, m, n, &alpha, d_A, m, d_x, 1, &beta, d_y, 1));
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("    elapsed time: %.5f (sec)\n", milliseconds / 1000.0);

	// Retrieve the output vector from the device
	CHECK_CUBLAS(cublasGetVector(m, sizeof(float), d_y, 1, y, 1));


	/**********************************************
	 *  Moltiplicazione matrix-matrix CUBLAS
	 **********************************************/

	printf("\n**  Matrix-Matrix product...\n");
  printf("    C(%d x %d) = A(%d x %d) * B(%d x %d)\n",m,p,m,n,n,p);

  //plot_mat_Col_Maj(m, n, A, 'A');
  //plot_mat_Col_Maj(n, p, B, 'B');

	CHECK(cudaMemset(d_C, 0,  m * p *sizeof(float)));
	CHECK(cudaEventRecord(start));
	CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, p, n, &alpha, d_A, m, d_B, n, &beta, d_C, m));
	CHECK(cudaEventRecord(stop));
	CHECK(cudaEventSynchronize(stop));
	CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
	printf("    elapsed time: %.5f (sec)\n", milliseconds / 1000.0);

	// Retrieve the output vector from the device
	CHECK_CUBLAS(cublasGetMatrix(m, p, sizeof(float), d_C, m, C, m));

  //plot_mat_Col_Maj(m, p, C, 'C');


	/*****************************************************
	 *  Moltiplicazione matrix-matrix kernel ad-hoc
	 *****************************************************/

	printf("\n**  Matrix-Matrix product using ad-hoc kernel (with SMEM)...\n");
  printf("    C(%d x %d) = A(%d x %d) * B(%d x %d)\n",m,p,m,n,n,p);
  
  float *A1, *B1; 
  srand(10);
	generate_random_dense_matrix_Row_Maj(m, n, &A1);
	generate_random_dense_matrix_Row_Maj(n, p, &B1);

  //plot_mat_Row_Maj(m, n, A1, 'A');
  //plot_mat_Row_Maj(n, p, B1, 'B');

	// copy matrices A and B to the GPU
	CHECK(cudaMemcpy(d_A, A1, m * n * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_B, B1, n * p * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemset(d_C, 0.0f, m * p * sizeof(float)));

	// grid block dims = shared mem dims = BLOCK_SIZE
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((p + block.x - 1) / block.x, (m + block.y - 1) / block.y);
	CHECK(cudaEventRecord(start));
	matProdSMEMstatic<<<grid, block>>>(d_A, d_B, d_C, n, m, p);
  CHECK(cudaDeviceSynchronize());
	CHECK(cudaEventRecord(stop));
	CHECK(cudaEventSynchronize(stop));
	CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
	printf("    elapsed time: %.5f (sec)\n", milliseconds / 1000.0);

	// copy the array 'C' back from the GPU to the CPU
	CHECK(cudaMemcpy(C, d_C, m * p * sizeof(float), cudaMemcpyDeviceToHost));

  //plot_mat_Row_Maj(m, p, C, 'C');
  
	// free memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaFree(d_x);
	cudaFree(d_y);
	CHECK_CUBLAS(cublasDestroy(handle));

	return EXIT_SUCCESS;
}


/*
 * Generate a vector of length N with random single-precision floating-point
 * values between 0 and 100.
 */

void generate_random_vector(int n, float **x) {
	float *z = (float *) malloc(sizeof(float) * n);

	for (int i = 0; i < n; i++)
		z[i] = (float)rand() / RAND_MAX;
	*x = z;
}

/*
 * Generate a matrix with M rows and N columns in column-major order. The matrix
 * will be filled with random single-precision floating-point values between 0 and 10
 */
void generate_random_dense_matrix_Col_Maj(int rows, int cols, float **A) {
	float *a = (float *) malloc(sizeof(float) * rows * cols);

  float val = 1.0;
  for (int c = 0; c < cols; ++c)
    for (int r = 0; r < rows; ++r){
      a[IDX2C(r,c,rows)] = val;
      val += 1;
    }
	*A = a;
}

void generate_random_dense_matrix_Row_Maj(int rows, int cols, float **A) {
	float *a = (float *) malloc(sizeof(float) * rows * cols);

  float val = 1.0;
	for (int r = 0; r < rows; r++)
		for (int c = 0; c < cols; c++) {
			a[IDX2R(r,c,cols)] = val;
      val += 1;
		}
	*A = a;
}

void plot_mat_Row_Maj(int rows, int cols, float *A, char name) {
  printf("\nShow mat %c...\n", name);
	for(int r = 0; r < rows; ++r){
    for(int c = 0; c < cols; ++c)
			printf("%4.1f ", A[IDX2R(r,c,cols)]);
    printf("\n");
	} 
  printf("\n");
}

void plot_mat_Col_Maj(int rows, int cols, float *A, char name) {
  printf("\nShow mat %c...\n", name);
  for(int r = 0; r < rows; ++r){
    for(int c = 0; c < cols; ++c)
      printf("%4.1f ", A[IDX2C(r,c,rows)]);
    printf("\n");
  }
  printf("\n");
}


/*
 * Kernel for matrix product with static SMEM
 *      C   =   A   *   B
 *   (m x p) (m x n) (n x p)
 */
__global__ void matProdSMEMstatic(float* A, float* B, float* C, int n, int m, int p) {
	// indexes
	uint row = blockIdx.y * blockDim.y + threadIdx.y; // in [0..m]
	uint col = blockIdx.x * blockDim.x + threadIdx.x; // in [0..p]

	// target: compute the right sum for the given row and col
	float sum = 0.0;

	// static shared memory
	__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

	// loop over blocks from block row of matrix A
	// and block column of matrix B
	uint numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

  if (threadIdx.y == 0 && threadIdx.x == 0) {
    //printf("n = %d.  m = %d.   p = %d\n", m,n,p);
    
  }

	for (uint i = 0; i < numBlocks; i++) {

		// copy block from matrix to shared memory
		uint r = i * BLOCK_SIZE + threadIdx.y;
		uint c = i * BLOCK_SIZE + threadIdx.x;
		As[threadIdx.y][threadIdx.x] = A[IDX2R(row, c, n)];
		Bs[threadIdx.y][threadIdx.x] = B[IDX2R(r, col, p)];

		__syncthreads();  //  BARRIER SYNC on SMEM loading

		uint K = BLOCK_SIZE;
		if (i == (numBlocks - 1)) 
      K = n - i * BLOCK_SIZE;   // tune last block

		// compute this part of row-column product
		for (uint k = 0; k < K; k++)
			sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

		__syncthreads();  //  BARRIER SYNC on prod over blocks
	}

	// store computed element in matrix C
	if (row < m && col < p) {
		C[row * p + col] = sum;
    //printf("C[%d] = %f   -   row = %d, col = %d \n",row * p + col, C[row * p + col],row,col );
  }
}

