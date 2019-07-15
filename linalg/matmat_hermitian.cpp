#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <complex.h>
#include <stdio.h>
#include <string.h>
#include "stopwatch.h"

extern "C" void chemm_(char *side, char *uplo, int *M, int *N, void *alpha, void *A, int *lda, void *B, int *ldb, void *beta, void *C, int *ldc);

int main(int argc, char **argv)
{
	int M = 256;	//Number of rows of C
	int N = 512;	//Number of columns of C
	printf("Matrix A is %d x %d\n", M, M);
	printf("Matrix B is %d x %d\n", M, N);
	printf("Matrix C is %d x %d\n", M, N);

	char side = 'L'; //The operation is C = alpha * A * B + beta * C
	char uplo = 'U'; //We use the upper triangular portion of A

	std::complex<float> alpha = 1.0 + 0.5 * I;
	std::complex<float> beta = 0.25;

	int lda = M, ldb = M, ldc = M;

	std::complex<float> *A = (std::complex<float>*)malloc(sizeof(std::complex<float>) * M * M);
	std::complex<float> *B = (std::complex<float>*)malloc(sizeof(std::complex<float>) * M * N);
	std::complex<float> *C = (std::complex<float>*)malloc(sizeof(std::complex<float>) * M * N);

	//Initialize with random data
	printf("Initializing with random data...\n"); fflush(stdout);

	long seed = 5724985725;
	boost::mt19937 eng(seed);
	boost::uniform_real<double> udist(0.0, 1.0);
	boost::variate_generator<boost::mt19937&, boost::uniform_real<double>> rng(eng, udist);
	for (long i = 0; i < (long)M * M; i++)
		A[i] = rng() + rng() * I;
	for (long i = 0; i < (long)M * N; i++) {
		B[i] = rng() + rng() * I;
		C[i] = rng() + rng() * I;
	}

	printf("Performing BLAS operation CHEMM: C = alpha * A * B + beta * C\n"); fflush(stdout);

	Stopwatch s = Stopwatch();
	stopwatchStart(&s);

	chemm_(&side, &uplo, &M, &N, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);

	stopwatchStop(&s);

	printf("Elapsed Time: %5.6f sec\n", s.elapsedTime);

	free(A); free(B); free(C);
}
