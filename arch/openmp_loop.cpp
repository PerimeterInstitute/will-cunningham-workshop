#include <omp.h>
#include <stdio.h>
#include <string.h>
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

	Stopwatch s = Stopwatch();
	stopwatchStart(&s);

	#pragma omp parallel for
	for (long i = 0; i < N; i++)
		z[i] = x[i] + y[i];

	stopwatchStop(&s);

	printf("Elapsed Time: %5.6f sec\n", s.elapsedTime);
	printf("Efficiency:   %5.6f GFLOPS\n", (double)N / (1000000000 * s.elapsedTime));
	printf("Threads Used: %d\n", omp_get_max_threads());

	return 0;
}
