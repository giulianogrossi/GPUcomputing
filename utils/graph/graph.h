#include <random>

#ifndef GRAPH_H
#define GRAPH_H

typedef unsigned int node;     // graph node
typedef unsigned int node_sz;  // graph node size

/**
 * Base structure (array 1D format) of a graph
 */
struct GraphStruct {
	node_sz nodeSize{0};             // num of graph nodes
	node_sz edgeSize{0};             // num of graph edges
	node_sz* cumDegs{nullptr};       // cumsum of node degrees
	node* neighs{nullptr};           // list of neighbors for all nodes (edges)

	~GraphStruct() {delete[] neighs; delete[] cumDegs;}

	// check whether node j is a neighbor of node i
	bool isNeighbor(node i, node j) {
		for (unsigned k = 0; k < deg(i); k++) 
			if (neighs[cumDegs[i]+k] == j)
	    	return true;
	  return false;
	}

	// return the degree of node i
	node_sz deg(node i) {
		return(cumDegs[i+1]-cumDegs[i]);
	}

};

/**
 * It manages a graph for CPU & GPU
 */
class Graph {
	float density{0.0f};	        // Probability of an edge (Erdos graph)
	GraphStruct * str{nullptr};     // graph structure
	node_sz maxDeg{0};
	node_sz minDeg{0};
	float meanDeg{0.0f};
	bool connected{true};
	bool GPUEnabled{true};

public:
	Graph(node_sz nn, bool GPUEnb) : GPUEnabled{GPUEnb} {setup(nn);}
	void setup(node_sz);	                             // CPU/GPU mem setup
	void randGraph(float, std::default_random_engine&);  // generate an Erdos random graph
	void print(bool);
	void print_d(GraphStruct *, bool);
	GraphStruct* getStruct() {return str;}
	void memsetGPU(node_sz, std::string);                 // use UVA memory on CPU/GPU
};

#endif
