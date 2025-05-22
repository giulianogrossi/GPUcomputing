#pragma once

#include <curand_kernel.h>
#include "../utils/graph/graph.h"
#include "../utils/common.h"

/**
 *  graph coloring struct (colors are: 1,2,3,..,k)
 */

struct Coloring {
	bool		uncoloredNodes;
	uint		numOfColors;
	uint	*	coloring;   // each element denotes a color
};

Coloring* LubyGreedy(GraphStruct*);
void printColoring (Coloring*, GraphStruct*, bool);
__global__ void LubyJPcolorer (Coloring*, GraphStruct*, uint*) ;
__global__ void init(uint seed, curandState_t*, uint*, uint);
__global__ void findIS (Coloring*, GraphStruct*, uint*);
__global__ void print_d(GraphStruct*, bool);