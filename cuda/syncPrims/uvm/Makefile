CUDA_DIR ?= /usr/local/cuda
CUDA_LIB_DIR := $(CUDA_DIR)/lib
ifeq ($(shell uname -m), x86_64)
     ifeq ($(shell if test -d $(CUDA_DIR)/lib64; then echo T; else echo F; fi), T)
		   CUDA_LIB_DIR := $(CUDA_DIR)/lib64
     endif
endif

# Volta+ parts with CUDA 11+ have nanosleep
HAS_NANOSLEEP ?= 0
CC := nvcc
INCLUDE := $(CUDA_DIR)/include
OPTS := -O3 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 $(CUDA_CPPFLAGS) -DHAS_NANOSLEEP=$(HAS_NANOSLEEP)
OPTS_DEBUG := -g -O0 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 $(CUDA_CPPFLAGS) -DHAS_NANOSLEEP=$(HAS_NANOSLEEP)

SRC = main.cu

EXE = allSyncPrims-1kernel

$(EXE): release

release: $(SRC)
	$(CC) $(OPTS) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lcudart

debug: $(SRC)
	$(CC) $(OPTS_DEBUG) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lcudart

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo *~
