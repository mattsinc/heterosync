#ifndef _GPUKERNEL_UTILS_CU
#define _GPUKERNEL_UTILS_CU

/*
  This file contains functions that are used by all/most of the GPU benchmarks
  we use in DeNovo.

  NOTE: All of these functions are dummy functions, they don't actually do 
  anything inside them.  The function calls will be intercepted by GPGPU-Sim,
  which will then call into GEMS as is appropriate.
*/
#define SPECIAL_REGION 65535
#define SCOPE_LOCAL_REGION 57526
#define READ_ONLY_REGION 5787
#define RELAX_ATOM_REGION (SCOPE_LOCAL_REGION + 1)

/*
  max exponential backoff value (need to make this desired
  power of 2 * 2 because we use bitwise ANDs of MAX_BACKOFF_EXP-1 to
  do the wraparound.
*/
#define MAX_BACKOFF_EXP 256

__device__ __forceinline__ void __gpuLock(unsigned int * lock_address)
{
  unsigned int returnVal = 0;
  do {
    returnVal = atomicCAS(lock_address, 0, 1);
    // acquire semantics needed here
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
    }
  } while (returnVal != 0);
}

__device__ __forceinline__ bool __gpuTryLock(unsigned int * lock_address)
{
  unsigned int locked = atomicCAS(lock_address, 0, 1);
  // acquire semantics needed here
  return locked==0;
}


__device__ __forceinline__ void __gpuUnlock(unsigned int * lock_address)
{
  __threadfence();
  atomicExch(lock_address, 0);
}

#endif // _GPUKERNEL_UTILS_CU
