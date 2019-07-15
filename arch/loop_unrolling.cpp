#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <stdio.h>
#include <string.h>
#include "stopwatch.h"

int main(int argc, char **argv)
{
	long N = 1000000000;
	long mask = 391759154;
	long *X = (long*)malloc(sizeof(long) * N);
	long *Y = (long*)malloc(sizeof(long) * N);

	if (X == NULL || Y == NULL) {
	  printf("Failed to allocate memory: X=%p, Y=%p, %s\n", X, Y, strerror(errno));
	  return -1;
	}

	//Initialize with random data
	printf("Initializing with random data...\n"); fflush(stdout);

	long seed = 5724985725;
	boost::mt19937 eng(seed);
	boost::uniform_real<double> udist(0.0, 1.0);
	boost::variate_generator<boost::mt19937&, boost::uniform_real<double>> rng(eng, udist);
	for (long i = 0; i < N; i++) {
		X[i] = rng() * LONG_MAX;
		Y[i] = rng() * LONG_MAX;
	}

	printf("Loop without unrolling:\n"); fflush(stdout);

	Stopwatch s = Stopwatch();
	stopwatchStart(&s);

	for (long i = 0; i < N; i++)
		X[i] ^= mask;

	stopwatchStop(&s);

	printf("Elapsed Time: %5.6f sec\n", s.elapsedTime); 
	printf("Loop with 4x unrolling:\n"); fflush(stdout);

	stopwatchReset(&s);
	stopwatchStart(&s);

	for (long i = 0; i < N; i += 4) {
		X[i] ^= mask;
		X[i+1] ^= mask;
		X[i+2] ^= mask;
		X[i+3] ^= mask;
	}

	stopwatchStop(&s);

	printf("ElapsedTime: %5.6f sec\n", s.elapsedTime); 
	printf("Loop with 8x unrolling:\n"); fflush(stdout);

	stopwatchReset(&s);
	stopwatchStart(&s);

	for (long i = 0; i < N; i += 8) {
		X[i] ^= mask;
		X[i+1] ^= mask;
		X[i+2] ^= mask;
		X[i+3] ^= mask;
		X[i+4] ^= mask;
		X[i+5] ^= mask;
		X[i+6] ^= mask;
		X[i+7] ^= mask;
	}

	stopwatchStop(&s);

	printf("ElapsedTime: %5.6f sec\n", s.elapsedTime); fflush(stdout);

	free(X); free(Y);
}
