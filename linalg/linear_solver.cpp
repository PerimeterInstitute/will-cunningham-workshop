#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <complex.h>
#include <stdio.h>
#include <string.h>
#include "stopwatch.h"

extern "C" void zcgesv_(int *N, int *nrhs, void *A, int *lda, int *ipiv, void *B, int *ldb, void *X, int *ldx, void *work, void *swork, double *rwork, int *iter, int *info);

int main(int argc, char **argv)
{
	int N = 512;	//Number of linear equations
	printf("Matrix is %d x %d\n", N, N);

	int nrhs = 16;	//Number of constraints (columns in X, B)
	printf("Using %d constraints\n", nrhs);
	int lda = N, ldx = N, ldb = N;
	int iter, info;

	std::complex<double> *A = (std::complex<double>*)malloc(sizeof(std::complex<double>) * N * N);
	std::complex<double> *X = (std::complex<double>*)malloc(sizeof(std::complex<double>) * N * nrhs);
	std::complex<double> *B = (std::complex<double>*)malloc(sizeof(std::complex<double>) * N * nrhs);

	std::complex<double> *work = (std::complex<double>*)malloc(sizeof(std::complex<double>) * N * nrhs);
	std::complex<float> *swork = (std::complex<float>*)malloc(sizeof(std::complex<float>) * N * (nrhs + N));
	double *rwork = (double*)malloc(sizeof(double) * N);

	int *ipiv = (int*)malloc(sizeof(int) * N);

	//Initialize with random data
	printf("Initializing with random data...\n"); fflush(stdout);

	long seed = 5724985725;
	boost::mt19937 eng(seed);
	boost::uniform_real<double> udist(0.0, 1.0);
	boost::variate_generator<boost::mt19937&, boost::uniform_real<double>> rng(eng, udist);
	for (int i = 0; i < N * nrhs; i++)
		B[i] = rng() + rng() * I;

	for (long i = 0; i < (long)N * N; i++)
		A[i] = rng() + rng() * I;

	printf("Performing LAPACK operation ZCGESV: x = A^(-1)b\n"); fflush(stdout);

	Stopwatch s = Stopwatch();
	stopwatchStart(&s);

	zcgesv_(&N, &nrhs, A, &lda, ipiv, B, &ldb, X, &ldx, work, swork, rwork, &iter, &info);

	stopwatchStop(&s);

	printf("Elapsed Time: %5.6f sec\n", s.elapsedTime);

	if (info == 0)
		printf("Successful exit from ZCGESV.\n");
	else if (info > 0)
		printf("Matrix is singular!\n");
	else
		printf("%d-th argument had an illegal value!\n", -info);

	if (iter > 0)
		printf("Iterative refinement used %d iterations.\n", iter);
	else
		printf("Iterative refinement failed with error code %d\n", iter);

	printf("Solution stored in 'X'\n");
	printf("Residual vectors are stored in 'work'\n");

	free(A); free(X); free(B);
	free(work); free(swork); free(rwork);
	free(ipiv);
}
