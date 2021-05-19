
#include <stdio.h>
#include <stdlib.h>
#include "../../utils/graph/graph.h"
#include "../../utils/common.h"

__global__ void print_d(GraphStruct*, bool);

__global__ void initBFS(GraphStruct *G, bool *Fa, bool *Xa, unsigned int n) {

	int nodeID = threadIdx.x + blockIdx.x * blockDim.x;

	if (nodeID > n)
		return;
  
  // set Fa and Xa vectors to false
  Fa[nodeID] = false;
  Xa[nodeID] = false;
	
}

/**
 * Kernel: The BFS frontier corresponds to all the nodes
 *         being processed at the current level
 */
__global__ void cudaBFS(GraphStruct *G, bool *Fa, bool *Xa, int *Ca, bool *done, int n) {

	int nodeID = threadIdx.x + blockIdx.x * blockDim.x;   // node ID

	if (nodeID > n)
		return;

	if (Fa[nodeID]) {
		*done = false;
		Fa[nodeID] = false;
		Xa[nodeID] = true;
		int deg = G->cumDegs[nodeID + 1] - G->cumDegs[nodeID];
		int start = G->cumDegs[nodeID];
		for (int i = 0; i < deg; i++) {
			int neighID = G->neighs[start + i];
			if ( !Xa[neighID] ) {
				Ca[neighID] = Ca[nodeID] + 1;
				Fa[neighID] = true;
			}
		}
	}
}

/**
 * MAIN: BFS test both CPU & GPU
 */
int main() {

	int BLOCK_SIZE = 512;
	unsigned int N = 20;                // number of nodes for random graphs
	float prob = .5;                    // density (percentage) for random graphs
	std::default_random_engine eng{0};  // fixed seed
	bool GPUEnabled = 1;
	Graph graph(N,GPUEnabled);

	// generate a random graph
	graph.randGraph(prob,eng);
	
	printf("** Graph done! \n");

	// get the graph struct
	GraphStruct *G = graph.getStruct();
	print_d<<<1,1>>>(G, 1);

	// setup vars for BFS
	bool *Fa, *Va, *Xa;
	int *Ca;
	CHECK(cudaMallocManaged((void **)&Fa, N* sizeof(bool)));
	CHECK(cudaMallocManaged((void **)&Va, N* sizeof(bool)));
	CHECK(cudaMallocManaged((void **)&Xa, N* sizeof(bool)));
	CHECK(cudaMallocManaged((void **)&Ca, N* sizeof(int)));

	// set the source
	int source = 0;
	Fa[source] = true;

	bool done;
	bool *d_done;
	cudaMalloc((void **)&d_done, sizeof(bool));
	int count = 0;
	printf("** BFS array size N = %d\n",N);

	// events to measure time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int nThreads = N;
	dim3 block(min(nThreads, BLOCK_SIZE));
	dim3 grid((nThreads + block.x - 1) / block.x);
	cudaEventRecord(start);
	initBFS<<<grid, block>>>(G, Fa, Xa, N);
	CHECK(cudaDeviceSynchronize());
	Fa[0] = true;
	Ca[0] = 0;
	do {
		count++;
		done = true;
		cudaMemcpy(d_done, &done, sizeof(bool), cudaMemcpyHostToDevice);
		cudaBFS<<<grid, block>>>(G, Fa, Xa, Ca, d_done, N);
		cudaMemcpy(&done, d_done, sizeof(bool), cudaMemcpyDeviceToHost);
	} while (!done);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaEventRecord(stop));
	CHECK(cudaEventSynchronize(stop));
	float milliseconds;
	cudaEventElapsedTime(&milliseconds, start, stop);
	float GPUtime = milliseconds / 1000.0;
	printf("   elapsed time:   %.5f (sec)\n", GPUtime);
	printf("   Number of times the kernel is called : %d \n", count);

	int max_Ca = 0;
	printf("\nCost: ");
	for (int i = 0; i < N; i++) {
		if (Ca[i] > max_Ca)
			max_Ca = Ca[i];
	}
	printf("max Ca = %d\n", max_Ca);

	return 0;
}


