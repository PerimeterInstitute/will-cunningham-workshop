#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <omp.h>
#include <mkl.h>
#include <mkl_lapacke.h>
#include <stdio.h>
#include <string.h>
#include "stopwatch.h"

//NOTE: If you get a linker error when compiling a MKL program with GCC/G++,
//check out the Intel MKL Link Line Advisor (google it) for help

int main(int argc, char **argv)
{
	//mkl_set_num_threads(1);
	mkl_set_num_threads(omp_get_max_threads());

	MKL_INT M = 1024; //Number of data samples (rows)
	static const MKL_INT N = 4; //Number of unknown coefficients (columns)
	int nrhs = 1; //Number of functions we are fitting to

	int matrix_layout = LAPACK_COL_MAJOR;
	char trans = 'N'; //Not using a transposed matrix
	int lda = M;
	int ldb = std::max(M, N);

	//The function we will use is b = alpha * A
	//In this sense, alpha is the slope, A is the independent data
	//and b is the 'measured' data.
	float alpha[N] = { 0.5, 1.8, 0.7, -1.2 };

	float *A = (float*)malloc(sizeof(float) * lda * N);
	float *b = (float*)malloc(sizeof(float) * ldb * nrhs);

	//Initialize with random data
	printf("Initializing with random data...\n"); fflush(stdout);

	long seed = 5724985725;
	boost::mt19937 eng(seed);
	boost::uniform_real<double> udist(0.0, 1.0);
	float noise = 0.1;
	boost::normal_distribution<double> ndist(0.0, noise);
	boost::variate_generator<boost::mt19937&, boost::uniform_real<double>> urng(eng, udist);
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<double>> nrng(eng, ndist);

	for (unsigned i = 0; i < M * N; i++)
		A[i] = urng();
	for (unsigned i = 0; i < M; i++) {
		float sum = 0.0;
		for (unsigned j = 0; j < N; j++)
			sum += alpha[j] * A[j*M+i];
		b[i] = sum;

		//Add noise to the simulated data
		b[i] += nrng();
	}

	printf("Expected solution: (%f, %f, %f, %f)\n", alpha[0], alpha[1], alpha[2], alpha[3]);
	printf("Using Gaussian noise with sigma=%f\n", noise);
	printf("Performing MKL operation SGELS: min|b-A*x|\n"); fflush(stdout);

	Stopwatch s = Stopwatch();
	stopwatchStart(&s);

	lapack_int info = LAPACKE_sgels(matrix_layout, trans, M, N, nrhs, A, lda, b, ldb);

	stopwatchStop(&s);

	printf("Elapsed Time: %5.6f sec\n", s.elapsedTime);

	if (info == 0)
		printf("Successful exit from SGELS.\n");
	else if (info > 0)
		printf("Matrix is singular!\n");
	else
		printf("%d-th argument had an illegal value!\n", -info);

	printf("Found solution:\n");
	printf("x = (%f, %f, %f, %f)\n", b[0], b[1], b[2], b[3]);

	free(A); free(b);
}
