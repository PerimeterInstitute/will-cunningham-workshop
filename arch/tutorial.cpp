#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <float.h>
#include <stdio.h>
#include <string.h>
#include "stopwatch.h"

struct Node {
	float x;
	float y;
};

inline float distance(Node *nodes, unsigned i, unsigned j)
{
	if (i == j) return 0.0;

	float dx = fabs(nodes[i].x - nodes[j].x);
	float dy = fabs(nodes[i].y - nodes[j].y);
	float dist = sqrt(dx * dx + dy * dy);

	return dist;
}

int main(int argc, char **argv)
{
	//Graph properties
	unsigned N = 4096;
	printf("Nodes: %u\n", N);
	bool *adj = (bool*)malloc(sizeof(bool) * N * N);
	Node *nodes = (Node*)malloc(sizeof(Node) * N);

	//Initialize random number generator
	long seed = 5724985725;	//Change me to get unique results
	printf("Seed: %lu\n", seed);
	boost::mt19937 eng(seed);
	boost::uniform_real<double> udist(0.0, 1.0);
	boost::variate_generator<boost::mt19937&, boost::uniform_real<double>> rng(eng, udist);

	//Sample coordinates
	float xmin = 0.0, xmax = 1.0;
	float ymin = 0.0, ymax = 1.0;
	printf("Coordinate Domain: [%f, %f) x [%f, %f)\n", xmin, xmax, ymin, ymax);
	for (unsigned i = 0; i < N; i++) {
		nodes[i].x = rng() * (xmax - xmin) + xmin;
		nodes[i].y = rng() * (ymax - ymin) + ymin;
	}

	//Construct the graph
	float connection_threshold = 3.0 / sqrt((float)N);
	printf("Connection Radius: %.6f\n", connection_threshold);
	memset(adj, 0, N * N);
	for (unsigned i = 0; i < N; i++) {
		for (unsigned j = 0; j < N; j++) {
			float dist = distance(nodes, i, j);

			if (dist < connection_threshold) {
				adj[i*N+j] = true;
				adj[j*N+i] = true;
			}
		}
	}

	//Print the edge list to file
	FILE *f = fopen("rgg_edge_list.txt", "w");
	if (f == NULL)
		printf("Could not open file!\n");
	for (unsigned i = 0; i < N; i++)
		for (unsigned j = i + 1; j < N; j++)
			if (adj[i*N+j])
				fprintf(f, "%u %u\n", i, j);
	fclose(f);

	//Measure the average degree
	unsigned sum = 0;
	for (unsigned i = 0; i < N; i++)
		for (unsigned j = 0; j < N; j++)
			if (adj[i*N+j])
				sum += 1;
	float average_degree = (float)sum / N;
	printf("Average Degree: %.6f\n", average_degree);

	//Measure the clustering
	//We will come back to this in the next lecture
	
	//Greedy Routing
	unsigned npaths = N;
	unsigned success = 0;
	unsigned path_length = 0;
	bool *visited = (bool*)malloc(sizeof(bool) * N);
	for (unsigned i = 0; i < npaths; i++) {
		unsigned source = rng() * N;
		unsigned target = rng() * (N - 1);
		target += (unsigned)(source == target);

		unsigned current = source;
		unsigned hops = 0;
		memset(visited, 0, N);

		while (current != target) {
			visited[current] = true;

			//Check neighbors
			float min_dist = FLT_MAX;
			int next = -1;
			for (unsigned j = 0; j < N; j++) {
				if (adj[current*N+j]) {
					float dist = distance(nodes, j, target);
					if (dist < min_dist) {
						min_dist = dist;
						next = j;
					}
				}
			}

			if (next == -1) //This node has no neighbors (failure)
				break;

			if (visited[next]) //The next choice was already visited (failure)
				break;

			current = next; //Update the current location
			hops++;
		}

		if (current == target) {
			success++;
			path_length += hops;
		}
	}

	float success_ratio = (float)success / npaths;
	printf("Success Ratio: %.6f\n", success_ratio);
	printf("Mean Path Length: %.6f\n", (float)path_length / success);

	free(adj); free(nodes); free(visited);
}
