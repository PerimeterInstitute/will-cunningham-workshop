all: cache_access loop_unrolling branch_elimination tutorial

loop_unrolling:
	g++ -o loop_unrolling stopwatch.cpp loop_unrolling.cpp -O0

cache_access:
	g++ -o cache_access stopwatch.cpp cache_access.cpp -O0

branch_elimination:
	g++ -o branch_elimination stopwatch.cpp branch_elimination.cpp -O0

tutorial:
	g++ -o tutorial stopwatch.cpp tutorial.cpp -O0

clean:
	@rm -f loop_unrolling cache_access branch_elimination tutorial
