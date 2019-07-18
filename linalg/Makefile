all: matvec_general matvec_hermitian matmat_hermitian linear_solver LU_decomposition singular_value_decomposition geometric_rotation spectral_gap rgg_spectrum linear_least_squares rgg_spectrum_mkl tutorial

CFLAGS ?= $(shell pkg-config --cflags eigen3) -g
INTEL_HOME ?= /opt/intel
MKLROOT ?= /opt/intel/mkl

matvec_general:
	g++ -o matvec_general stopwatch.cpp matvec_general.cpp $(CFLAGS) -lblas

matvec_hermitian:
	g++ -o matvec_hermitian stopwatch.cpp matvec_hermitian.cpp $(CFLAGS) -lblas

matmat_hermitian:
	g++ -o matmat_hermitian stopwatch.cpp matmat_hermitian.cpp $(CFLAGS) -lblas

linear_solver:
	g++ -o linear_solver stopwatch.cpp linear_solver.cpp $(CFLAGS) -llapack -lblas

LU_decomposition:
	g++ -o LU_decomposition stopwatch.cpp LU_decomposition.cpp $(CFLAGS) -llapack -lblas

singular_value_decomposition:
	g++ -o singular_value_decomposition stopwatch.cpp singular_value_decomposition.cpp $(CFLAGS) -llapack -lblas

geometric_rotation:
	g++ -o geometric_rotation stopwatch.cpp geometric_rotation.cpp $(CFLAGS)

spectral_gap:
	g++ -o spectral_gap stopwatch.cpp spectral_gap.cpp $(CFLAGS) -fopenmp -O3

rgg_spectrum:
	g++ -o rgg_spectrum stopwatch.cpp rgg_spectrum.cpp $(CFLAGS) -fopenmp -O3

linear_least_squares:
	g++ -o linear_least_squares stopwatch.cpp linear_least_squares.cpp -fopenmp -O3 -march=core-avx2 -mavx2 -frename-registers -fno-signed-zeros -I $(INTEL_HOME)/mkl/include -Wl,--start-group $(INTEL_HOME)/mkl/lib/intel64/libmkl_intel_ilp64.a $(INTEL_HOME)/mkl/lib/intel64/libmkl_gnu_thread.a $(INTEL_HOME)/mkl/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl

rgg_spectrum_mkl:
	g++ -g -o rgg_spectrum_mkl stopwatch.cpp rgg_spectrum_mkl.cpp -fopenmp -O3 -DMKL_ILP64 -m64 -I $(MKLROOT)/include -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_ilp64.a $(MKLROOT)/lib/intel64/libmkl_gnu_thread.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl
	@#icpc -o rgg_spectrum_mkl stopwatch.cpp rgg_spectrum_mkl.cpp -O3 -DMKL_ILP64 -I $(MKLROOT)/include -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_ilp64.a $(MKLROOT)/lib/intel64/libmkl_intel_thread.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl 

tutorial:
	g++ -o tutorial stopwatch.cpp tutorial.cpp -O3 $(CFLAGS) -llapack -lblas

clean:
	@rm -f matvec_general matvec_hermitian matmat_hermitian linear_solver LU_decomposition singular_value_decomposition geometric_rotation spectral_gap rgg_spectrum linear_least_squares rgg_spectrum_mkl tutorial
