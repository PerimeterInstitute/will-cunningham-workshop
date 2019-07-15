#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <stdio.h>
#include <string.h>
#include "stopwatch.h"

extern "C" void sgemv_(char *trans, int *M, int *N, float *alpha, float *A, int *lda, float *X, int *incx, float *beta, float *Y, int *incy);

int main(int argc, char **argv)
{
	int M = 512;	//Rows in matrix
	int N = 512;	//Columns in matrix
	printf("Rows: %d\n", M);
	printf("Columns: %d\n", N);
	char trans = 'N'; //Operation is y = alpha * A * x + beta * y

	float alpha = 2.0, beta = 0.7;
	float *A = (float*)malloc(sizeof(float) * M * N);
	int lda = M;	//Leading dimension for the full matrix
			//is simply the number of rows
	int incx = 1, incy = 1;	//Increment by 1 (use every element of X, Y)
	float *X = (float*)malloc(sizeof(float) * N);
	float *Y = (float*)malloc(sizeof(float) * M);

	//Initialize with random data
	printf("Initializing with random data...\n"); fflush(stdout);

	long seed = 5724985725;
	boost::mt19937 eng(seed);
	boost::uniform_real<double> udist(0.0, 1.0);
	boost::variate_generator<boost::mt19937&, boost::uniform_real<double>> rng(eng, udist);
	for (int i = 0; i < N; i++)
		X[i] = rng();
	for (int i = 0; i < M; i++)
		Y[i] = rng();
	for (long i = 0; i < (long)M * N; i++)
		A[i] = rng();

	printf("Performing BLAS operation SGEMV: y = alpha * A * x + beta * y\n"); fflush(stdout);

	Stopwatch s = Stopwatch();
	stopwatchStart(&s);

	sgemv_(&trans, &M, &N, &alpha, A, &lda, X, &incx, &beta, Y, &incy);

	stopwatchStop(&s);

	printf("Elapsed Time: %5.6f sec\n", s.elapsedTime);

	free(A); free(X); free(Y);
}
