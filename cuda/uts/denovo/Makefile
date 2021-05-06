CUDA_DIR ?= /usr/local/cuda
CUDA_LIB_DIR := $(CUDA_DIR)/lib
ifeq ($(shell uname -m), x86_64)
     ifeq ($(shell if test -d $(CUDA_DIR)/lib64; then echo T; else echo F; fi), T)
	   CUDA_LIB_DIR := $(CUDA_DIR)/lib64
     endif
endif

CC := nvcc
INCLUDE := $(CUDA_DIR)/include
OPTS := -O3 -arch sm_21
OPTS_DEBUG := -g -O0 -arch sm_21

SRC = uts.cu

EXE = uts

#$(EXE): release
#
#release: $(SRC)
#	$(CC) $(OPTS) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lcudart

$(EXE): $(SRC)
	$(CC) $(OPTS) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lcudart

debug: $(SRC)
	$(CC) $(OPTS_DEBUG) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lcudart

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo *~

## All of these files will get .o files made out of them (the .cu.cpp file
## will get the special annotations as seen by the 'cuda' label)
#SRCS = uts.cu.cpp
## All of these will be removed when make clean is called
#uts: uts.cu treebar.h gpuKernels_util.cu
#	nvcc  -o uts uts.cu -lcudart
#	#nvcc -arch sm_20 -w -cuda uts.cu -keep
