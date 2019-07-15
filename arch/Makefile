all: cache_access loop_unrolling branch_elimination tutorial

CFLAGS ?= -O0 -g

loop_unrolling:
	g++ -o loop_unrolling stopwatch.cpp loop_unrolling.cpp $(CFLAGS)

cache_access:
	g++ -o cache_access stopwatch.cpp cache_access.cpp $(CFLAGS)

branch_elimination:
	g++ -o branch_elimination stopwatch.cpp branch_elimination.cpp $(CFLAGS)

tutorial:
	g++ -o tutorial stopwatch.cpp tutorial.cpp $(CFLAGS)

clean:
	@rm -f loop_unrolling cache_access branch_elimination tutorial
