#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <stdio.h>
#include <string.h>
#include "stopwatch.h"

int main(int argc, char **argv)
{
	Eigen::Matrix3f m; //The 3d rotation matrix
	Eigen::Vector3f v; //The vector we will rotate
	Eigen::Vector3f axis; //The axis about which we rotate
	float angle = M_PI / 4.0; //The angle by which we rotate

	//Initialize with random data
	printf("Initializing with random data...\n"); fflush(stdout);

	long seed = 5724985725;
	boost::mt19937 eng(seed);
	boost::uniform_real<double> udist(0.0, 1.0);
	boost::variate_generator<boost::mt19937&, boost::uniform_real<double>> rng(eng, udist);
	v << rng(), rng(), rng();
	axis << rng(), rng(), rng();
	axis.normalize();

	printf("Rotating vector about the z-axis.\n");

	Stopwatch s = Stopwatch();
	stopwatchStart(&s);

	m = Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitZ());
	v = m * v;

	stopwatchStop(&s);
	printf("Elapsed Time: %5.6f sec\n", s.elapsedTime);
	printf("Rotating vector about a custom axis.\n");

	stopwatchReset(&s);
	stopwatchStart(&s);

	m = Eigen::AngleAxisf(angle, axis);
	v = m * v;

	stopwatchStop(&s);
	printf("Elapsed Time: %5.6f sec\n", s.elapsedTime);

	//Access vector elements with v.x(), v.y(), and v.z()
	//Access matrix elements with m(i,j)
}
