#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <float.h>
#include <stdio.h>
#include <string.h>
#include "stopwatch.h"

extern "C" void sgesvd_(char *jobu, char *jobvt, int *M, int *N, float *A, int *lda, float *S, float *U, int *ldu, float *VT, int *ldvt, float *work, int *lwork, int *info);

//To increase dimension, you need to add more coordinates to each Node
struct Node {
	float x;
	float y;
};

//This also gets updated for higher dimension
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
	unsigned N = 1024;
	printf("Nodes: %u\n", N);
	float *adj = (float*)calloc(N * N, sizeof(float));
	Node *nodes = (Node*)malloc(sizeof(Node) * N);

	//Initialize random number generator
	//long seed = 5724985725;	//Change me to get unique results
	long seed = (long)time(NULL);
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
	for (unsigned i = 0; i < N; i++) {
		for (unsigned j = 0; j < N; j++) {
			float dist = distance(nodes, i, j);

			if (dist < connection_threshold) {
				adj[i*N+j] = 1.0;
				adj[j*N+i] = 1.0;
			}
		}
	}

	//Print the edge list to file
	/*FILE *f = fopen("rgg_edge_list.txt", "w");
	if (f == NULL)
		printf("Could not open file!\n");
	for (unsigned i = 0; i < N; i++)
		for (unsigned j = i + 1; j < N; j++)
			if (adj[i*N+j])
				fprintf(f, "%u %u\n", i, j);
	fclose(f);*/

	//SVD
	char jobu = 'N';
	char jobvt = 'N';
	int lda = N, ldu = N, ldvt = N, lwork = 5 * N;
	int info;

	float *S = (float*)malloc(sizeof(float) * N);
	float *work = (float*)malloc(sizeof(float) * lwork);

	printf("Calculating singular values..."); fflush(stdout);

	sgesvd_(&jobu, &jobvt, (int*)&N, (int*)&N, adj, &lda, S, NULL, &ldu, NULL, &ldvt, work, &lwork, &info);

	printf("completed\n");

	if (info == 0)
		printf("Successful exit from ZCGESV.\n");
	else if (info > 0)
		printf("There were %d superdiagonals which did not converge.\n", info);
	else
		printf("%d-th argument had an illegal value!\n", -info);

	//Print the singular values to file (append)
	FILE *f = fopen("rgg_singular_values.dat", "a");
	if (f == NULL)
		printf("Could not open file!\n");
	for (unsigned i = 0; i < N; i++)
		fprintf(f, "%.8f ", S[i]);
	fprintf(f, "\n");
	fclose(f);

	free(adj); free(nodes); free(S); free(work);
}
