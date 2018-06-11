#include "cudaLocks.h"

cudaError_t cudaLocksInit(const int maxBlocksPerKernel, const int numMutexes,
                          const int numSemaphores, 
                          const bool pageAlign, /* const region_t locksReg*/
                          const int NUM_SM)
{
  if (maxBlocksPerKernel <= 0)    return cudaErrorInitializationError;
  if (numMutexes <= 0)            return cudaErrorInitializationError;
  if (numSemaphores <= 0)         return cudaErrorInitializationError;

  cudaLockData_t * cpuLockData_temp = (cudaLockData_t *)malloc(sizeof(cudaLockData_t) + 0x1000);
  if (pageAlign) {
    cpuLockData = (cudaLockData_t *)(((((unsigned long long)cpuLockData_temp) >> 12) << 12) + 0x1000);
  } else {
    cpuLockData = cpuLockData_temp;
  }

  // initialize some of the lock data's values
  cpuLockData->maxBufferSize          = maxBlocksPerKernel;
  cpuLockData->arrayStride            = (maxBlocksPerKernel + NUM_SM) / 16 * 16;
  cpuLockData->mutexCount             = numMutexes;
  cpuLockData->semaphoreCount         = numSemaphores;

  // malloc arrays for the lock data structure
  unsigned int * barrierBuffers_temp = (unsigned int *)malloc((sizeof(unsigned int) * cpuLockData->arrayStride * 2) + 0x1000);
  int * mutexBuffers_temp = (int *)malloc((sizeof(int) * cpuLockData->arrayStride * cpuLockData->mutexCount) + 0x1000);
  unsigned int * mutexBufferHeads_temp = (unsigned int *)malloc((sizeof(unsigned int) * cpuLockData->mutexCount) + 0x1000);
  unsigned int * mutexBufferTails_temp = (unsigned int *)malloc((sizeof(unsigned int) * cpuLockData->mutexCount) + 0x1000);

  unsigned int * semaphoreBuffers_temp = (unsigned int *)malloc((sizeof(unsigned int) * 4 * cpuLockData->semaphoreCount) + 0x1000);

  if (pageAlign) {
    cpuLockData->barrierBuffers = (unsigned int *)(((((unsigned long long)barrierBuffers_temp) >> 12) << 12) + 0x1000);
    cpuLockData->mutexBuffers = (int *)(((((unsigned long long)mutexBuffers_temp) >> 12) << 12) + 0x1000);
    cpuLockData->mutexBufferHeads = (unsigned int *)(((((unsigned long long)mutexBufferHeads_temp) >> 12) << 12) + 0x1000);
    cpuLockData->mutexBufferTails = (unsigned int *)(((((unsigned long long)mutexBufferTails_temp) >> 12) << 12) + 0x1000);
    cpuLockData->semaphoreBuffers = (unsigned int *)(((((unsigned long long)semaphoreBuffers_temp) >> 12) << 12) + 0x1000);
  } else {
    cpuLockData->barrierBuffers = barrierBuffers_temp;
    cpuLockData->mutexBuffers = mutexBuffers_temp;
    cpuLockData->mutexBufferHeads = mutexBufferHeads_temp;
    cpuLockData->mutexBufferTails = mutexBufferTails_temp;
    cpuLockData->semaphoreBuffers = semaphoreBuffers_temp;
  }

  // initialize all memory
  int i = 0;
  for (i = 0; i < (cpuLockData->arrayStride * 2); ++i) {
    cpuLockData->barrierBuffers[i] = 0;
  }
  for (i = 0; i < (cpuLockData->arrayStride * cpuLockData->mutexCount); ++i) {
    // set the first location for each SM to 1 so that the ring buffer can be
    // used by the first TB right away (otherwise livelock because no locations
    // ever == 1)
    if (i % cpuLockData->arrayStride == 0) { cpuLockData->mutexBuffers[i] = 1; }
    // for all other locations initialize to -1 so TBs for these locations
    // don't think it's their turn right away
    else { cpuLockData->mutexBuffers[i] = -1; }
  }
  for (i = 0; i < cpuLockData->mutexCount; ++i) {
    cpuLockData->mutexBufferHeads[i] = 0;
    cpuLockData->mutexBufferTails[i] = 0;
  }
  for (i = 0; i < (cpuLockData->semaphoreCount * 4); ++i) {
    cpuLockData->semaphoreBuffers[i] = 0;
  }

  return cudaSuccess;
}

cudaError_t cudaLocksDestroy()
{
  if (cpuLockData == NULL) { return cudaErrorInitializationError; }
  free(cpuLockData->mutexBuffers);
  free(cpuLockData->mutexBufferHeads);
  free(cpuLockData->mutexBufferTails);

  free(cpuLockData->semaphoreBuffers);

  free(cpuLockData);

  return cudaSuccess;
}
