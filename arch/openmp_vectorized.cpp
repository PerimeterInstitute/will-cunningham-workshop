#include <stdio.h>
#include <string.h>
#include <x86intrin.h>
#include "stopwatch.h"

int main(int argc, char **argv)
{
	long N = 1000000000;
	double *x = (double*)malloc(sizeof(double) * N);
	double *y = (double*)malloc(sizeof(double) * N);
	double *z = (double*)malloc(sizeof(double) * N);

	memset(x, 1.0, sizeof(double) * N);
	memset(y, 2.0, sizeof(double) * N);
	memset(z, 0.0, sizeof(double) * N);

	double total = 0.0;
	for (long i = 0; i < N; i++) {
		z[i] = x[i] * y[i];
		total += z[i];
	}

	Stopwatch s = Stopwatch();
	stopwatchStart(&s);

	__m256d sum = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
	#pragma omp parallel for
	for (long i = 0; i < N; i += 4) {
		__m256d data0 = _mm256_loadu_pd(&x[i]);
		__m256d data1 = _mm256_loadu_pd(&y[i]);
		data0 = _mm256_mul_pd(data0, data1);
		#pragma omp critical
		sum = _mm256_add_pd(sum, data0);
	}
	_mm256_storeu_pd(&z[0], sum);
	total = z[0] + z[1] + z[2] + z[3];

	stopwatchStop(&s);

	printf("Elapsed Time: %5.6f sec\n", s.elapsedTime);
	printf("Efficiency:   %5.6f GFLOPS\n", (double)N / (1000000000 * s.elapsedTime));

	return 0;
}
