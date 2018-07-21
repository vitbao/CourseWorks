# Warnings
WFLAGS := -Wall -Wextra -Wsign-conversion -Wsign-compare

# Optimization and architecture
OPT  := -O3 -mavx -fopenmp

# Language standard
CCSTD	:= -std=c99
CXXSTD	:= -std=c++14

# Linker options
LDFLAGS := -fopenmp

# Names of executables to create
EXEC := neuralnet neuralnetOMP neuralnetOMP1 neuralnet_verify neuralnetOMP_verify neuralnetOMP1_verify 

.DEFAULT_GOAL := all

.PHONY: debug
debug : OPT  := -O0 -g -fno-omit-frame-pointer -fsanitize=address
debug : LDFLAGS := -fsanitize=address
debug : $(EXEC)

all : Makefile $(EXEC)

neuralnet: neuralnet.c
	gcc -o neuralnet neuralnet.c -lm
neuralnetOMP: neuralnetOMP.c
	gcc -o neuralnetOMP neuralnetOMP.c -lm -fopenmp
neuralnetOMP1: neuralnetOMP1.c
	gcc -o neuralnetOMP1 neuralnetOMP1.c -lm -fopenmp
neuralnet_verify: neuralnet_verify.c
	gcc -o neuralnet_verify neuralnet_verify.c -lm
neuralnetOMP_verify: neuralnetOMP_verify.c
	gcc -o neuralnetOMP_verify neuralnetOMP_verify.c -lm -fopenmp
neuralnetOMP1_verify: neuralnetOMP1_verify.c
	gcc -o neuralnetOMP1_verify neuralnetOMP1_verify.c -lm -fopenmp

.PHONY: clean
clean:
	@ rm -f *.o $(EXEC) 
