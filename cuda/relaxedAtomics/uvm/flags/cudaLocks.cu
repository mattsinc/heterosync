#include "cudaLocks.h"

cudaError_t cudaLocksInit(const int maxBlocksPerKernel, const bool pageAlign)
{
  cudaError_t cudaErr = cudaGetLastError();
  checkError(cudaErr, "Start cudaLocksInit");

  cudaMallocHost(&cpuLockData, sizeof(cudaLockData_t));

  if (maxBlocksPerKernel <= 0)    return cudaErrorInitializationError;

  // initialize some of the lock data's values
  cpuLockData->maxBufferSize          = maxBlocksPerKernel;
  cpuLockData->arrayStride            = (maxBlocksPerKernel + NUM_SM) / 16 * 16;

  cudaMalloc(&cpuLockData->barrierBuffers,   sizeof(unsigned int) * cpuLockData->arrayStride * 2);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaErr = cudaGetLastError();
  checkError(cudaErr, "Before memset");

  cudaDeviceSynchronize();
  cudaEventRecord(start, 0);

  cudaMemset(cpuLockData->barrierBuffers, 0, sizeof(unsigned int) * cpuLockData->arrayStride * 2);

  cudaDeviceSynchronize();
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  float elapsedTime = 0.0f;
  cudaEventElapsedTime(&elapsedTime, start, end);
  fprintf(stdout, "\tmemcpy H->D 1 elapsed time: %f ms\n", elapsedTime);
  fflush(stdout);

  cudaEventDestroy(start);
  cudaEventDestroy(end);

  return cudaSuccess;
}

cudaError_t cudaLocksDestroy()
{
  if (cpuLockData == NULL) { return cudaErrorInitializationError; }

  cudaFree(cpuLockData->barrierBuffers);
  cudaFreeHost(cpuLockData);

  return cudaSuccess;
}
