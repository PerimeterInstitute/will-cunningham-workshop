#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include "stopwatch.h"

int main(int argc, char **argv)
{
	long N = 10000;
	double *adj = (double*)malloc(sizeof(double) * N * N);
	memset(adj, 0.0, sizeof(double) * N * N);

	bool use_omp = true;

	long npairs = N * (N - 1) / 2;

	Stopwatch s = Stopwatch();
	stopwatchStart(&s);

	long seed = 9325723957;
	#pragma omp parallel if (use_omp)
	{
		boost::mt19937 eng(seed ^ omp_get_thread_num());
		boost::uniform_real<double> udist(0.0, 1.0);
		boost::variate_generator<boost::mt19937&, boost::uniform_real<double> > rng(eng, udist);

		/*#pragma omp for
		for (long i = 0; i < N; i++)
			for (long j = i + 1; j < N; j++)
				if (rng() < 0.5)
					adj[i*N+j] = 1.0;*/

		#pragma omp for
		for (long k = 0; k < npairs; k++) {
			int i = (int)(k / (N - 1));
			int j = (int)(k % (N - 1) + 1);
			int do_map = i >= j;
			if (j < N >> 1) {
				i += do_map * ((((N >> 1) - i) << 1) - 1);
				j += do_map * (((N >> 1) - j) << 1);
			}

			if (rng() < 0.5)
				adj[i*N+j] = 1.0;
		}
	}

	stopwatchStop(&s);

	printf("Elapsed Time: %5.6f sec\n", s.elapsedTime);
	printf("Efficiency:   %5.6f GFLOPS\n", (double)N * (N - 1) / (2.0 * 1000000000 * s.elapsedTime));
	if (use_omp)
		printf("Threads Used: %d\n", omp_get_max_threads());

	return 0;
}
