#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <stdio.h>
#include <string.h>
#include "stopwatch.h"

extern "C" void sgetrf_(int *M, int *N, float *A, int *lda, int *ipiv, int *info);

int main(int argc, char **argv)
{
	int M = 1024;	//Number of rows of A
	int N = 1024;	//Number of columns of A
	printf("Matrix is %d x %d\n", M, N);

	int lda = M;
	int info;

	float *A = (float*)malloc(sizeof(float) * M * N);
	int *ipiv = (int*)malloc(sizeof(int) * std::min(M, N));

	//Initialize with random data
	printf("Initializing with random data...\n"); fflush(stdout);

	long seed = 5724985725;
	boost::mt19937 eng(seed);
	boost::uniform_real<double> udist(0.0, 1.0);
	boost::variate_generator<boost::mt19937&, boost::uniform_real<double>> rng(eng, udist);
	for (long i = 0; i < (long)M * N; i++)
		A[i] = rng();

	printf("Performing LAPACK operation SGETRF: LU decomposition of A\n"); fflush(stdout);

	Stopwatch s = Stopwatch();
	stopwatchStart(&s);

	sgetrf_(&M, &N, A, &lda, ipiv, &info);

	stopwatchStop(&s);

	printf("Elapsed Time: %5.6f sec\n", s.elapsedTime);

	if (info == 0)
		printf("Successful exit from ZCGESV.\n");
	else if (info > 0)
		printf("Matrix is singular!\n");
	else
		printf("%d-th argument had an illegal value!\n", -info);

	printf("Solution stored in 'A'\n");
	printf("Row permutation vector stored in 'ipiv'\n");

	free(A); free(ipiv);
}
