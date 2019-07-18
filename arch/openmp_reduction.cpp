#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
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

	bool use_omp = true;

	Stopwatch s = Stopwatch();
	stopwatchStart(&s);

	long seed = 9325723957;
	#pragma omp parallel if (use_omp)
	{
		boost::mt19937 eng(seed ^ omp_get_thread_num());
		boost::uniform_real<double> udist(0.0, 1.0);
		boost::variate_generator<boost::mt19937&, boost::uniform_real<double> > rng(eng, udist);
		#pragma omp for
		for (long i = 0; i < N; i++) {
			x[i] = rng() * 0.5;
			y[i] = rng() * 0.5;
		}
	}

	stopwatchStop(&s);

	long counter = 0;
	#pragma omp parallel for reduction (+ : counter) if (use_omp)
	for (long i = 0; i < N; i++)
		if (x[i] + y[i] < 0.5)
			counter++;

	printf("Elapsed Time: %5.6f sec\n", s.elapsedTime);
	printf("Efficiency:   %5.6f GFLOPS\n", (double)N / (1000000000 * s.elapsedTime));
	if (use_omp)
		printf("Threads Used: %d\n", omp_get_max_threads());

	return 0;
}
