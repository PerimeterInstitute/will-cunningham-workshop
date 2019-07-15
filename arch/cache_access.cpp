#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <stdio.h>
#include <string.h>
#include "stopwatch.h"

int main(int argc, char **argv)
{
	long N = 8000;
	long mask = 391759154;
	long *X = (long*)malloc(sizeof(long) * N * N);
	long *Y = (long*)malloc(sizeof(long) * N * N);

	//Initialize with random data
	printf("Initializing with random data...\n"); fflush(stdout);

	long seed = 5724985725;
	boost::mt19937 eng(seed);
	boost::uniform_real<double> udist(0.0, 1.0);
	boost::variate_generator<boost::mt19937&, boost::uniform_real<double>> rng(eng, udist);
	for (long i = 0; i < N * N; i++) {
		X[i] = rng() * LONG_MAX;
		Y[i] = rng() * LONG_MAX;
	}

	printf("Column-Major Access:\n"); fflush(stdout);

	Stopwatch s = Stopwatch();
	stopwatchStart(&s);

	for (long i = 0; i < N; i++)
		for (long j = 0; j < N; j++)
			X[j*N+i] += mask;

	stopwatchStop(&s);

	printf("Elapsed Time: %5.6f sec\n", s.elapsedTime);
	printf("Row-Major Access:\n"); fflush(stdout);

	stopwatchReset(&s);
	stopwatchStart(&s);

	for (long i = 0; i < N; i++)
		for (long j = 0; j < N; j++)
			Y[i*N+j] += mask;

	stopwatchStop(&s);
	printf("Elapsed Time: %5.6f sec\n", s.elapsedTime); fflush(stdout);

	free(X); free(Y);
}
