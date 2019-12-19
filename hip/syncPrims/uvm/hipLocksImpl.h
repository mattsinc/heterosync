#include "hipLocks.h"

hipError_t hipLocksInit(const int maxBlocksPerKernel, const int numMutexes,
                          const int numSemaphores,
                          const bool pageAlign, /* const region_t locksReg*/
                          const int NUM_SM)
{
  hipError_t hipErr = hipGetLastError();
  checkError(hipErr, "Start hipLocksInit");

  hipHostMalloc(&cpuLockData, sizeof(hipLockData_t));

  if (maxBlocksPerKernel <= 0)    return hipErrorInitializationError;
  if (numMutexes <= 0)            return hipErrorInitializationError;
  if (numSemaphores <= 0)         return hipErrorInitializationError;

  // initialize some of the lock data's values
  cpuLockData->maxBufferSize          = maxBlocksPerKernel;
  cpuLockData->arrayStride            = (maxBlocksPerKernel + NUM_SM) /
                                        NUM_WORDS_PER_CACHELINE * NUM_WORDS_PER_CACHELINE;
  cpuLockData->mutexCount             = numMutexes;
  cpuLockData->semaphoreCount         = numSemaphores;

  hipMalloc(&cpuLockData->barrierBuffers,   sizeof(unsigned int) * cpuLockData->arrayStride * 2);

  hipMalloc(&cpuLockData->mutexBuffers,     sizeof(int) * cpuLockData->arrayStride * cpuLockData->mutexCount);
  hipMalloc(&cpuLockData->mutexBufferHeads, sizeof(unsigned int) * cpuLockData->mutexCount);
  hipMalloc(&cpuLockData->mutexBufferTails, sizeof(unsigned int) * cpuLockData->mutexCount);

  hipMalloc(&cpuLockData->semaphoreBuffers, sizeof(unsigned int) * 4 * cpuLockData->semaphoreCount);

  hipEvent_t start, end;
  hipEventCreate(&start);
  hipEventCreate(&end);

  hipErr = hipGetLastError();
  checkError(hipErr, "Before memset");

  hipDeviceSynchronize();
  hipEventRecord(start, 0);

  hipMemset(cpuLockData->barrierBuffers, 0,
            sizeof(unsigned int) * cpuLockData->arrayStride * 2);

  hipMemset(cpuLockData->mutexBufferHeads, 0,
            sizeof(unsigned int) * cpuLockData->mutexCount);
  hipMemset(cpuLockData->mutexBufferTails, 0,
            sizeof(unsigned int) * cpuLockData->mutexCount);

  /*
    initialize mutexBuffers to appropriate values

    set the first location for each SM to 1 so that the ring buffer can be
    used by the first TB right away (otherwise livelock because no locations
    ever == 1)

    for all other locations initialize to -1 so TBs for these locations
    don't think it's their turn right away
  */
  for (int i = 0; i < (cpuLockData->arrayStride * cpuLockData->mutexCount);
       i += cpuLockData->arrayStride) {
    hipMemset(&(cpuLockData->mutexBuffers[i]), 1, sizeof(int));
    hipMemset(&(cpuLockData->mutexBuffers[i + 1]), -1,
              (cpuLockData->arrayStride - 1) * sizeof(int));
  }

  hipMemset(cpuLockData->semaphoreBuffers, 0,
            sizeof(unsigned int) * cpuLockData->semaphoreCount * 4);

  hipDeviceSynchronize();
  hipEventRecord(end, 0);
  hipEventSynchronize(end);
  float elapsedTime = 0.0f;
  hipEventElapsedTime(&elapsedTime, start, end);
  fprintf(stdout, "\tmemcpy H->D 1 elapsed time: %f ms\n", elapsedTime);
  fflush(stdout);

  hipEventDestroy(start);
  hipEventDestroy(end);

  return hipSuccess;
}

hipError_t hipLocksDestroy()
{
  if (cpuLockData == NULL) { return hipErrorInitializationError; }
  hipFree(cpuLockData->mutexBuffers);
  hipFree(cpuLockData->mutexBufferHeads);
  hipFree(cpuLockData->mutexBufferTails);

  hipFree(cpuLockData->semaphoreBuffers);

  hipHostFree(cpuLockData);

  return hipSuccess;
}
