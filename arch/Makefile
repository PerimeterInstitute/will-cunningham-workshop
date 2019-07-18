all: cache_access loop_unrolling branch_elimination openmp_loop openmp_double_loop openmp_reduction openmp_vectorized tutorial

CFLAGS ?= -O0 -g

loop_unrolling:
	g++ -o loop_unrolling stopwatch.cpp loop_unrolling.cpp $(CFLAGS)

cache_access:
	g++ -o cache_access stopwatch.cpp cache_access.cpp $(CFLAGS)

branch_elimination:
	g++ -o branch_elimination stopwatch.cpp branch_elimination.cpp $(CFLAGS)

openmp_loop:
	g++ -o openmp_loop stopwatch.cpp openmp_loop.cpp -O0 -fopenmp

openmp_double_loop:
	g++ -o openmp_double_loop stopwatch.cpp openmp_double_loop.cpp -O0 -fopenmp

openmp_reduction:
	g++ -o openmp_reduction stopwatch.cpp openmp_reduction.cpp -O0 -fopenmp

openmp_vectorized:
	g++ -o openmp_vectorized stopwatch.cpp openmp_vectorized.cpp -O0 -fopenmp -mavx2 -march=core-avx2

tutorial:
	g++ -o tutorial stopwatch.cpp tutorial.cpp $(CFLAGS)

clean:
	@rm -f loop_unrolling cache_access branch_elimination openmp_loop openmp_double_loop openmp_reduction openmp_vectorized tutorial
