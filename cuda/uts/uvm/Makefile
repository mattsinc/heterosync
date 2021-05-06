CUDA_DIR ?= /usr/local/cuda
CUDA_LIB_DIR := $(CUDA_DIR)/lib
ifeq ($(shell uname -m), x86_64)
     ifeq ($(shell if test -d $(CUDA_DIR)/lib64; then echo T; else echo F; fi), T)
	   CUDA_LIB_DIR := $(CUDA_DIR)/lib64
     endif
endif

CC := nvcc
INCLUDE := $(CUDA_DIR)/include
GENCODE := -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70
OPTS := -O3 $(GENCODE) $(CUDA_CPPFLAGS)
OPTS_DEBUG := -G -g -O0 $(GENCODE) $(CUDA_CPPFLAGS)

SRC = uts.cu

EXE = uts

$(EXE): release

release: $(SRC)
	$(CC) $(OPTS) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lcudart

debug: $(SRC)
	$(CC) $(OPTS_DEBUG) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lcudart

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo *~
