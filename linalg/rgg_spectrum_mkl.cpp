#include <algorithm>
#include <omp.h>
#include <math.h>
#include <mkl.h>
#include <stdio.h>
#include <string.h>
#include "stopwatch.h"

//static const int NUM_THREADS = omp_get_max_threads();
static const int NUM_THREADS = 1;

int main(int argc, char **argv)
{
	//Read the data
	FILE *f  = fopen("./rgg_edge_list.txt", "r");
	if (f == NULL)
		printf("Could not open file!\n");

	char line[64];
	char delimeters[] = " \t";

	//First pass: determine matrix size
	MKL_INT N = 0;
	long N_edges = 0;
	while (fgets(line, sizeof(line), f) != NULL) {
		unsigned i = atoi(strtok(line, delimeters));
		unsigned j = atoi(strtok(NULL, delimeters));

		N = std::max(N, (MKL_INT)std::max(i, j));
		N_edges++;
	}
	N++; //Because we use zero-based indexing
	fseek(f, 0, SEEK_SET);

	printf("Number of nodes: [%u]\n", (unsigned)N);
	fflush(stdout);

	mkl_set_num_threads(NUM_THREADS);
	printf("Filling matrix..."); fflush(stdout);

	MKL_INT alignment = 64;
	MKL_INT m0 = N, m = 0;
	float *A = (float*)mkl_calloc(N * N, sizeof(float), alignment);
	MKL_INT *degrees = (MKL_INT*)mkl_calloc(N, sizeof(MKL_INT), alignment);
	float *residuals = (float*)mkl_calloc(m0, sizeof(float), alignment);
	float *eigenvalues = (float*)mkl_calloc(m0, sizeof(float), alignment);
	float *eigenvectors = (float*)mkl_calloc(N * m0, sizeof(float), alignment);

	char uplo = 'F'; //Store the full matrix
	while (fgets(line, sizeof(line), f) != NULL) {
		unsigned i = atoi(strtok(line, delimeters));
		unsigned j = atoi(strtok(NULL, delimeters));
		A[i*N+j] = 1.0;
		A[j*N+i] = 1.0;
		degrees[i]++;
		degrees[j]++;
	}
	fclose(f);
	printf("completed.\n");
	printf("Calculating the eigenvalues and eigenvectors of A...\n"); fflush(stdout);

	MKL_INT fpm[128];
	MKL_INT lda = N;
	float emin = -10, emax = 40;
	float epsout;
	MKL_INT loop = 0, info = 0;

	memset(fpm, 0, sizeof(MKL_INT) * 128);
	fpm[0] = 1;

	Stopwatch s = Stopwatch();
	stopwatchStart(&s);

	feastinit(fpm);
	sfeast_syev(&uplo, &N, A, &lda, fpm, &epsout, &loop, &emin, &emax, &m0, eigenvalues, eigenvectors, &m, residuals, &info);

	stopwatchStop(&s);

	if (info == 0) {
		printf("Elapsed Time: %5.6f sec\n", s.elapsedTime);
		printf("Execution successful.\n");
	} else {
		printf("Execution returned error code [%d]\n", (int)info);
		return info;
	}
	
	printf("Used [%d] refinement loops.\n", (int)loop);
	printf("We found [%d] eigenvalues in the interval [%f, %f].\n", (int)m, emin, emax);
	fflush(stdout);

	printf("Printing eigenvalues to file...\n");
	f = fopen("./rgg_spectrum_mkl.txt", "w");
	if (f == NULL)
		printf("Could not open file!\n");
	for (unsigned i = 0; i < m; i++)
		fprintf(f, "%.8f\n", eigenvalues[i]);
	fclose(f);
	printf("Completed.\n"); fflush(stdout);

	//Calculate the clustering coefficient
	double num_triangles = 0.0;
	for (unsigned i = 0; i < N; i++)
		num_triangles += pow(eigenvalues[i], 3.0);
	num_triangles /= 6.0;
	double num_connected_triples = 0.0;
	for (unsigned i = 0; i < N; i++)
		num_connected_triples += degrees[i] * (degrees[i] - 1.0);
	num_connected_triples /= 2.0;
	printf("Mean Number of 3-Cycles (Triangles): %f\n", num_triangles);
	printf("Clustering: %f\n", 3.0 * num_triangles / num_connected_triples);

	mkl_free(A); mkl_free(degrees); mkl_free(residuals);
	mkl_free(eigenvalues); mkl_free(eigenvectors);
}
