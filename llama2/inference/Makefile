# choose your compiler, e.g. gcc/clang
# example override to clang: make build CC=clang 
# eg: CC=/usr/local/opt/llvm/bin/clang
CC = gcc
BIN_DIR = ./bin

init:
	@mkdir -p ${BIN_DIR}

# use math lib
# the most basic way of building that is most likely to work on most systems
.PHONY: build 
build: init main.c
	$(CC) -std=c11 -O3 -o ${BIN_DIR}/inference main.c -lm

# useful for a debug build, can then e.g. analyze with valgrind, example:
# $ valgrind --leak-check=full ./inference out/model.bin -n 3
.PHONY: build_debug
build_debug: init main.c
	$(CC) -g -std=c11 -o ${BIN_DIR}/inference main.c -lm

# https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
# https://simonbyrne.github.io/notes/fastmath/
# -Ofast enables all -O3 optimizations.
# Disregards strict standards compliance.
# It also enables optimizations that are not valid for all standard-compliant programs.
# It turns on -ffast-math, -fallow-store-data-races and the Fortran-specific
# -fstack-arrays, unless -fmax-stack-var-size is specified, and -fno-protect-parens.
# It turns off -fsemantic-interposition.
# In our specific application this is *probably* okay to use
.PHONY: build_fast
build_fast: init main.c
	$(CC) -std=c11 -Ofast -o ${BIN_DIR}/inference main.c -lm

# additionally compiles with OpenMP, allowing multithreaded builds
# make sure to also enable multiple threads when buildning, e.g.:
# OMP_NUM_THREADS=4 ./inference out/model.bin
.PHONY: build_omp
build_omp: init main.c
	$(CC) -std=c11 -Ofast -fopenmp -march=native main.c  -lm  -o ${BIN_DIR}/inference

clean:
	@rm -rf ${BIN_DIR}/*