#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <stdio.h>
#include <string.h>
#include "stopwatch.h"

int main(int argc, char **argv)
{
	long N = 100000000;
	float sum = 0.0;
	float *X = (float*)malloc(sizeof(float) * N);
	float *Y = (float*)malloc(sizeof(float) * N);

	//Initialize with random data
	printf("Initializing with random data...\n"); fflush(stdout);

	long seed = 5724985725;
	boost::mt19937 eng(seed);
	boost::uniform_real<double> udist(0.0, 1.0);
	boost::variate_generator<boost::mt19937&, boost::uniform_real<double>> rng(eng, udist);
	for (long i = 0; i < N; i++) {
		X[i] = (rng() * 2.0) - 1.0;
		Y[i] = (rng() * 2.0) - 1.0;
	}

	printf("Accumulation with Branch:\n"); fflush(stdout);

	Stopwatch s = Stopwatch();
	stopwatchStart(&s);

	for (long i = 0; i < N; i++)
		if (X[i] >= 0.0)
			sum += X[i];

	stopwatchStop(&s);

	printf("Elapsed Time: %5.6f sec\n", s.elapsedTime);
	printf("Accumulation without Branch:\n"); fflush(stdout);

	stopwatchReset(&s);
	sum = 0;

	stopwatchStart(&s);

	for (long i = 0; i < N; i++)
		sum -= (~((*(int*)&Y[i]) >> 31)) * Y[i];

	stopwatchStop(&s);
	printf("Elapsed Time: %5.6f sec\n", s.elapsedTime); fflush(stdout);

	free(X); free(Y);
}
