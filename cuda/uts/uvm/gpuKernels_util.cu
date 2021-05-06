#ifndef _GPUKERNEL_UTILS_CU
#define _GPUKERNEL_UTILS_CU

__device__ __forceinline__ void __gpuLock(unsigned int * lock_address)
{
  unsigned int returnVal = 0;
  do {
    returnVal = atomicCAS(lock_address, 0, 1);
    // acquire semantics needed here
    __threadfence();
  } while (returnVal != 0);
}

// Use an unpaired atomic to check if the lock is held first
__device__ __forceinline__ void __gpuLockRelaxed(unsigned int * lock_address)
{
  unsigned int returnVal = 0;
  do {
    if (atomicAdd(lock_address, 0) == 0) {
      returnVal = atomicCAS(lock_address, 0, 1);
      // acquire semantics needed here
      __threadfence();
    }
  } while (returnVal != 0);
}

__device__ __forceinline__ bool __gpuTryLock(unsigned int * lock_address)
{
  unsigned int locked = atomicCAS(lock_address, 0, 1);
  // acquire semantics needed here
  __threadfence();
  return locked==0;
}


__device__ __forceinline__ void __gpuUnlock(unsigned int * lock_address)
{
  __threadfence();
  atomicExch(lock_address, 0);
}

#endif // _GPUKERNEL_UTILS_CU
