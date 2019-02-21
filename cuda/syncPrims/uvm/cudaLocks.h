#ifndef __CUDALOCKS_CU__
#define __CUDALOCKS_CU__

typedef struct cudaLockData
{
  int maxBufferSize;
  int arrayStride;
  int mutexCount;
  int semaphoreCount;

  unsigned int * barrierBuffers;
  int * mutexBuffers;
  unsigned int * mutexBufferHeads;
  unsigned int * mutexBufferTails;
  unsigned int * semaphoreBuffers;
} cudaLockData_t;

typedef unsigned int cudaMutex_t;
typedef unsigned int cudaSemaphore_t;

static cudaLockData_t * cpuLockData;

cudaError_t cudaLocksInit(const int maxBlocksPerKernel, const int numMutexes,
                          const int numSemaphores, const bool pageAlign,
                          const int NUM_SM);
cudaError_t cudaLocksDestroy();

#endif
