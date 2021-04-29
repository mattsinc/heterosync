#ifndef __CUDALOCKS_CU__
#define __CUDALOCKS_CU__

typedef struct cudaLockData
{
  int maxBufferSize;
  int arrayStride;

  unsigned int * barrierBuffers;
} cudaLockData_t;

static cudaLockData_t * cpuLockData;

cudaError_t cudaLocksInit(const int maxBlocksPerKernel,
                          const bool pageAlign/*, const region_t locksReg*/);
cudaError_t cudaLocksDestroy();

#endif
