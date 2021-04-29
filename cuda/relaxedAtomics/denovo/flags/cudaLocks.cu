#include "cudaLocks.h"

cudaError_t cudaLocksInit(const int maxBlocksPerKernel,
                          const bool pageAlign/*, const region_t locksReg*/)
{
  if (maxBlocksPerKernel <= 0)    return cudaErrorInitializationError;

  cudaLockData_t * cpuLockData_temp = (cudaLockData_t *)malloc(sizeof(cudaLockData_t) + 0x1000/*, locksReg*/);
  if (pageAlign) {
    cpuLockData = (cudaLockData_t *)(((((unsigned long long)cpuLockData_temp) >> 12) << 12) + 0x1000);
  } else {
    cpuLockData = cpuLockData_temp;
  }

  // initialize some of the lock data's values
  cpuLockData->maxBufferSize          = maxBlocksPerKernel;
  cpuLockData->arrayStride            = (maxBlocksPerKernel + NUM_SM) / 16 * 16;

  // malloc arrays for the lock data structure
  unsigned int * barrierBuffers_temp = (unsigned int *)malloc((sizeof(unsigned int) * cpuLockData->arrayStride * 2) + 0x1000);

  if (pageAlign) {
    cpuLockData->barrierBuffers = (unsigned int *)(((((unsigned long long)barrierBuffers_temp) >> 12) << 12) + 0x1000);
  } else {
    cpuLockData->barrierBuffers = barrierBuffers_temp;
  }

  // initialize all memory
  int i = 0;
  for (i = 0; i < (cpuLockData->arrayStride * 2); ++i) {
    cpuLockData->barrierBuffers[i] = 0;
  }

  return cudaSuccess;
}

cudaError_t cudaLocksDestroy()
{
  if (cpuLockData == NULL) { return cudaErrorInitializationError; }
  free(cpuLockData->barrierBuffers);

  free(cpuLockData);

  return cudaSuccess;
}
