#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include "stopwatch.h"

//Compute the spectral gap of a large sparse matrix (smallest eigenvalue of a matrix)

static const int NUM_THREADS = omp_get_max_threads();

enum Method {
	CONJ_GRAD,
	BICGSTAB
};

//Choose one of the two options here:
//static const Method method = CONJ_GRAD;
static const Method method = BICGSTAB;

int main(int argc, char **argv)
{
	//Read the data
	FILE *f  = fopen("./sgap_edge_list.txt", "r");
	if (f == NULL)
		printf("Could not open file!\n");

	char line[64];
	char delimeters[] = " \t";

	//First pass: determine matrix size, number of non-zero entries, and degrees
	unsigned N = 0, N_edges = 0;
	while (fgets(line, sizeof(line), f) != NULL) {
		unsigned i = atoi(strtok(line, delimeters));
		unsigned j = atoi(strtok(NULL, delimeters));

		N = std::max(N, std::max(i, j));
		N_edges++;
	}
	N++; //Because we use zero-based indexing

	printf("Number of nodes: [%u]\n", N);
	printf("Number of edges: [%u]\n", N_edges);
	fflush(stdout);

	//The matrix we consider is A-2D
	//N_edg is the number of entries each in the upper triangle and lower triangle
	//N is the number of entries on the diagonal (due to D)
	unsigned nnz = 2 * N_edges + N;

	Eigen::setNbThreads(NUM_THREADS);
	printf("Filling sparse matrix..."); fflush(stdout);

	//The sparse matrix M = A - 2D
	Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> M(N, N);
	std::vector<Eigen::Triplet<double> > triples;
	Eigen::VectorXi degrees = Eigen::VectorXi::Zero(N);
	fseek(f, 0, SEEK_SET);
	for (unsigned k = 0; k < N_edges; k++) {
		assert(fgets(line, sizeof(line), f) != NULL);
		unsigned i = atoi(strtok(line, delimeters));
		unsigned j = atoi(strtok(NULL, delimeters));
		degrees[i]++;
		degrees[j]++;
		triples.push_back(Eigen::Triplet<double>(i, j, 1.0));
		triples.push_back(Eigen::Triplet<double>(j, i, 1.0));
	}
	fclose(f);

	for (unsigned k = 0; k < N; k++)
		triples.push_back(Eigen::Triplet<double>(k, k, -2.0 * degrees[k]));
	M.setFromTriplets(triples.begin(), triples.end());
	printf("completed.\n");

	long seed = 5724985725;
	boost::mt19937 eng(seed);
	boost::uniform_real<double> udist(0.0, 1.0);
	boost::variate_generator<boost::mt19937&, boost::uniform_real<double>> rng(eng, udist);

	//Initialize the eigenvector to random values
	Eigen::VectorXd eigenvector(N), x(N);
	for (int i = 0; i < N; i++)
		eigenvector[i] = rng();

	Eigen::VectorXd residual(N);
	int max_iter = 1000;	//Maximum iterations in power method
	double lambda = 0.0;	//The eigenvalue (spectral gap)
	double epsilon = 0.0;	//Final scalar residual error
	double tol = 1e-3;	//Convergence tolerance

	Stopwatch s = Stopwatch();
	stopwatchStart(&s);

	if (method == CONJ_GRAD) { //Use the conjugate gradient method to invert M
		printf("Initializing CG solver..."); fflush(stdout);
		Eigen::ConjugateGradient<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t>, Eigen::Lower | Eigen::Upper> cg;
		cg.compute(M);
		printf("completed.\n");
		
		for (int i = 0; i < max_iter; i++) {
			//Starting with M*x'=x; then x' = M^(-1)x
			//The first iteration this is just as good as starting with a random vector for x
			x = cg.solve(eigenvector).normalized();	//Gives x' using the inverse power method
			eigenvector = cg.solve(x);	//Gives M^{-1}*x'
			lambda = x.transpose() * eigenvector; //Gives (x')^T * M^{-1} * x'
			residual = eigenvector - lambda * x; //Gives the residual vector M^{-1} * x' - lambda * x'
			epsilon = residual.norm(); //Scalar residual
			printf("Eigenvalue: [%f]\tResidual: %.8f\n", lambda, epsilon); fflush(stdout);
			if (epsilon < tol) break;
		}
	} else if (method == BICGSTAB) { //Use the biconjugate gradient stabilized method to invert M
		printf("Initializing BiCGSTAB solver..."); fflush(stdout);
		Eigen::BiCGSTAB<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> > cg;
		cg.compute(M);
		printf("completed.\n");

		for (int i = 0; i < max_iter; i++) {
			//Starting with M*x'=x; then x' = M^(-1)x
			//The first iteration this is just as good as starting with a random vector for x
			x = cg.solve(eigenvector).normalized();	//Gives x' using the inverse power method
			eigenvector = cg.solve(x);	//Gives M^{-1}*x'
			lambda = x.transpose() * eigenvector; //Gives (x')^T * M^{-1} * x'
			residual = eigenvector - lambda * x; //Gives the residual vector M^{-1} * x' - lambda * x'
			epsilon = residual.norm(); //Scalar residual
			printf("Eigenvalue: [%f]\tResidual: %.8f\n", lambda, epsilon); fflush(stdout);
			if (epsilon < tol) break;
		}
	}

	stopwatchStop(&s);
	printf("Elapsed Time: %5.6f sec\n", s.elapsedTime);

	printf("Dominant Eigenvalue: %f\n", lambda);
	printf("Relative Residual Error: %f\n", epsilon);
}
