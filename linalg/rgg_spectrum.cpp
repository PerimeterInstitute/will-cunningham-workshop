#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include "stopwatch.h"

static const int NUM_THREADS = omp_get_max_threads();

int main(int argc, char **argv)
{
	//Read the data
	FILE *f  = fopen("./rgg_edge_list.txt", "r");
	if (f == NULL)
		printf("Could not open file!\n");

	char line[64];
	char delimeters[] = " \t";

	//First pass: determine matrix size
	unsigned N = 0;
	long N_edges = 0;
	while (fgets(line, sizeof(line), f) != NULL) {
		unsigned i = atoi(strtok(line, delimeters));
		unsigned j = atoi(strtok(NULL, delimeters));

		N = std::max(N, std::max(i, j));
		N_edges++;
	}
	N++; //Because we use zero-based indexing

	printf("Number of nodes: [%u]\n", N);
	fflush(stdout);

	Eigen::setNbThreads(NUM_THREADS);
	printf("Filling matrix..."); fflush(stdout);

	Eigen::MatrixXd A = Eigen::MatrixXd::Zero(N, N);
	Eigen::VectorXi degrees = Eigen::VectorXi::Zero(N);
	fseek(f, 0, SEEK_SET);
	while (fgets(line, sizeof(line), f) != NULL) {
		unsigned i = atoi(strtok(line, delimeters));
		unsigned j = atoi(strtok(NULL, delimeters));
		A(i, j) = 1;
		A(j, i) = 1;
		degrees(i)++;
		degrees(j)++;
	}
	fclose(f);
	printf("completed.\n");
	printf("Calculating the eigenvalues and eigenvectors of A...\n"); fflush(stdout);

	Stopwatch s = Stopwatch();
	stopwatchStart(&s);

	//The constructor will calculate the eigendecomposition
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(A);
	Eigen::VectorXd eigenvalues = eigensolver.eigenvalues();
	Eigen::MatrixXd eigenvectors = eigensolver.eigenvectors();

	stopwatchStop(&s);
	printf("Elapsed Time: %5.6f sec\n", s.elapsedTime);

	printf("Printing eigenvalues to file...\n");
	f = fopen("./rgg_spectrum.txt", "w");
	if (f == NULL)
		printf("Could not open file!\n");
	for (unsigned i = 0; i < N; i++)
		fprintf(f, "%.8f\n", eigenvalues(i));
	fclose(f);
	printf("Completed.\n"); fflush(stdout);

	//Calculate the clustering coefficient
	double num_triangles = 0.0;
	for (unsigned i = 0; i < N; i++)
		num_triangles += pow(eigenvalues(i), 3.0);
	num_triangles /= 6.0;
	double num_connected_triples = 0.0;
	for (unsigned i = 0; i < N; i++)
		num_connected_triples += degrees(i) * (degrees(i) - 1.0);
	num_connected_triples /= 2.0;
	printf("Mean Number of 3-Cycles (Triangles): %f\n", num_triangles);
	printf("Clustering: %f\n", 3.0 * num_triangles / num_connected_triples);
}
