#ifndef __HIPLOCKS_H__
#define __HIPLOCKS_H__

typedef struct hipLockData
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
} hipLockData_t;

typedef unsigned int hipMutex_t;
typedef unsigned int hipSemaphore_t;

static hipLockData_t * cpuLockData;

hipError_t hipLocksInit(const int maxBlocksPerKernel, const int numMutexes,
                        const int numSemaphores, const bool pageAlign,
                        const int NUM_SM);
hipError_t hipLocksDestroy();

#endif
