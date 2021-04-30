#ifndef __CUDALOCKS_CU__
#define __CUDALOCKS_CU__

#include "cuda_error.h"

typedef struct cudaLockData
{
  int maxBufferSize;
  int arrayStride;
  unsigned int * barrierBuffers;
} cudaLockData_t;

typedef unsigned int cudaMutex_t;
typedef unsigned int cudaSemaphore_t;

static cudaLockData_t * cpuLockData;

#endif
