#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <stdio.h>
#include <string.h>
#include "stopwatch.h"

extern "C" void sgesvd_(char *jobu, char *jobvt, int *M, int *N, float *A, int *lda, float *S, float *U, int *ldu, float *VT, int *ldvt, float *work, int *lwork, int *info);

int main(int argc, char **argv)
{
	int M = 256;	//Number of rows of A
	int N = 1024;	//Number of columns of A
	printf("Matrix is %d x %d\n", M, N);

	char jobu = 'A'; //Calculate all left singular vectors
	char jobvt = 'A'; //Calculate all right singular vectors
	int lda = M, ldu = M, ldvt = N, lwork = std::max(3 * std::min(M, N) + std::max(M, N), 5 * std::min(M, N));
	int info;

	float *A = (float*)malloc(sizeof(float) * M * N);
	float *S = (float*)malloc(sizeof(float) * std::min(M, N));
	float *U = (float*)malloc(sizeof(float) * M * M);
	float *VT = (float*)malloc(sizeof(float) * N * N);
	float *work = (float*)malloc(sizeof(float) * lwork);

	//Initialize with random data
	printf("Initializing with random data...\n"); fflush(stdout);

	long seed = 5724985725;
	boost::mt19937 eng(seed);
	boost::uniform_real<double> udist(0.0, 1.0);
	boost::variate_generator<boost::mt19937&, boost::uniform_real<double>> rng(eng, udist);
	for (long i = 0; i < (long)M * N; i++)
		A[i] = rng();

	printf("Performing LAPACK operation SGESVD: singular value decomposition of A\n"); fflush(stdout);

	Stopwatch s = Stopwatch();
	stopwatchStart(&s);

	sgesvd_(&jobu, &jobvt, &M, &N, A, &lda, S, U, &ldu, VT, &ldvt, work, &lwork, &info);

	stopwatchStop(&s);

	printf("Elapsed Time: %5.6f sec\n", s.elapsedTime);

	if (info == 0)
		printf("Successful exit from ZCGESV.\n");
	else if (info > 0)
		printf("There were %d superdiagonals which did not converge.\n", info);
	else
		printf("%d-th argument had an illegal value!\n", -info);

	printf("Solution stored in 'S'\n");
	printf("Left singular vectors stored in 'U'\n");
	printf("Right singular vectors (transposed) stored in 'VT'\n");

	free(A); free(S); free(U); free(VT); free(work);
}
