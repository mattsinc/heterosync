#ifndef __CUDALOCKS_CU__
#define __CUDALOCKS_CU__

// MDS: Reconstructed all structs from the way they're used in the code
typedef struct cudaLockData
{
  int maxBufferSize;
  int arrayStride;
  int mutexCount;
  int semaphoreCount;
  int conditionVariableCount;

  unsigned int * barrierBuffers;
  int * mutexBuffers;
  unsigned int * mutexBufferHeads;
  unsigned int * mutexBufferTails;
  unsigned int * semaphoreBuffers;
  int * conditionVariableBuffers;
  int * conditionVariableWaitBuffers;
  int * conditionVariableBufferHeads;
  int * conditionVariableBufferTails;
} cudaLockData_t;

typedef unsigned int cudaMutex_t;
typedef unsigned int cudaSemaphore_t;
typedef int cudaCondvar_t;

static cudaLockData_t * cpuLockData;

cudaError_t cudaLocksInit(const int maxBlocksPerKernel, const int numMutexes,
                          const int numSemaphores,
                          const int numConditionVariables);
cudaError_t cudaLocksDestroy();

#endif
