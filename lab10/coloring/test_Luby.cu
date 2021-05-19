
#include "coloring.h"

__global__ void print_d(GraphStruct*, bool);

int main(void) {
	unsigned int n = 10000;		        // number of nodes for random graphs
	float prob = .01;				    // density (percentage) for random graphs
	std::default_random_engine eng{0};  // fixed seed

	// new graph with n nodes
	Graph graph(n,1);

	// generate a random graph
	graph.randGraph(prob,eng);

	// get the graph struct
	GraphStruct *str = graph.getStruct();

	// print small graph
	if (n <= 20) {
		graph.print(true);  // CPU print
		print_d<<< 1, 1 >>>(str, true);  // GPU print
	}
	/* print example:
		** Graph (num node: 10, num edges: 46)
				 (min deg: 3, max deg: 7, mean deg: 4.6, connected: 1)
		   node(0)[5]-> 1 2 3 7 9
		   node(1)[3]-> 0 2 9
		   node(2)[5]-> 0 1 3 5 8
		   node(3)[7]-> 0 2 4 5 6 7 9
		   node(4)[5]-> 3 6 7 8 9
		   node(5)[5]-> 2 3 6 8 9
		   node(6)[4]-> 3 4 5 8
		   node(7)[3]-> 0 3 4
		   node(8)[4]-> 2 4 5 6
		   node(9)[5]-> 0 1 3 4 5
	   */

	// GPU Luby-JP greedy coloring
	Coloring* col = LubyGreedy(str);
	cudaDeviceSynchronize();
	printColoring(col, str, 1);

	/* Coloring example:
	 	** Graph (num node: 10, num edges: 36)
		** Coloring (num colors: 6)
		   color(1)-> 1 3 8
		   color(2)-> 0 6
		   color(3)-> 4 7
		   color(4)-> 9
		   color(5)-> 5
		   color(6)-> 2
	 */
	return EXIT_SUCCESS;
}
