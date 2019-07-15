#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <complex.h>
#include <stdio.h>
#include <string.h>
#include "stopwatch.h"

extern "C" void chpmv_(char *uplo, int *N, void *alpha, void *A, void *X, int *incx, void *beta, void *Y, int *incy);

int main(int argc, char **argv)
{
	int N = 512;	//Matrix dimension
	printf("Matrix is %d x %d\n", N, N);
	char uplo = 'U'; //We are specifying the upper triangular portion of A

	std::complex<float> alpha = 1.0 + 0.5 * I; //NOTE: complex.h reserves the variable I
	std::complex<float> beta = 3.0 * I;

	int incx = 1, incy = 1;

	//The packed format for triangular matrices uses column-major order
	//We have N(N+1)/2 entries, including the diagonal
	std::complex<float> *A = (std::complex<float>*)malloc(sizeof(std::complex<float>) * (N * (N + 1) / 2));
	std::complex<float> *X = (std::complex<float>*)malloc(sizeof(std::complex<float>) * N);
	std::complex<float> *Y = (std::complex<float>*)malloc(sizeof(std::complex<float>) * N);

	//Initialize with random data
	printf("Initializing with random data...\n"); fflush(stdout);

	long seed = 5724985725;
	boost::mt19937 eng(seed);
	boost::uniform_real<double> udist(0.0, 1.0);
	boost::variate_generator<boost::mt19937&, boost::uniform_real<double>> rng(eng, udist);
	for (int i = 0; i < N; i++) {
		X[i] = rng() + rng() * I;
		Y[i] = rng() + rng() * I;
	}

	for (long i = 0; i < (long)N * (N + 1) / 2; i++)
		A[i] = rng() + rng() * I;

	printf("Performing BLAS operation CHPMV: y = alpha * A * x + beta * y\n"); fflush(stdout);

	Stopwatch s = Stopwatch();
	stopwatchStart(&s);

	chpmv_(&uplo, &N, &alpha, A, X, &incx, &beta, Y, &incy);

	stopwatchStop(&s);

	printf("Elapsed Time: %5.6f sec\n", s.elapsedTime);

	free(A); free(X); free(Y);
}
