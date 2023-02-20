#include <stdio.h>
#include <iostream>
#include "graph.h"
#include "../common.h"

using namespace std;

/**
 * Set the CUDA Unified Memory for nodes and edges
 * @param memType node or edge memory type
 */
void Graph::memsetGPU(node_sz nn, string memType) {
	if (!memType.compare("nodes")) {
		CHECK(cudaMallocManaged(&str, sizeof(GraphStruct)));
		CHECK(cudaMallocManaged(&(str->cumDegs), (nn+1)*sizeof(node)));
	}
	else if (!memType.compare("edges")) {
		CHECK(cudaMallocManaged(&(str->neighs), str->edgeSize*sizeof(node)));
	}
}

/**
 * Print the graph on device (verbose = 1 for "verbose print")
 * @param verbose print the complete graph
 */
__global__ void print_d(GraphStruct* str, bool verbose) {
	printf("** Graph (num node: %d, num edges: %d)\n", str->nodeSize,str->edgeSize);

	if (verbose) {
		for (int i = 0; i < str->nodeSize; i++) {
			printf("  node(%d)[%d]-> ",i,str->cumDegs[i+1]-str->cumDegs[i]);
			for (int j = 0; j < str->cumDegs[i+1] - str->cumDegs[i]; j++) {
				printf("%d ", str->neighs[str->cumDegs[i]+j]);
			}
			printf("\n");
		}
		printf("\n");
	}
}


