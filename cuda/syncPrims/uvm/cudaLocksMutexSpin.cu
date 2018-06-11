#ifndef __CUDALOCKSMUTEXSPIN_H__
#define __CUDALOCKSMUTEXSPIN_H__

inline __host__ cudaError_t cudaMutexCreateSpin(cudaMutex_t * const handle,
                                                const int mutexNumber)
{
  *handle = mutexNumber;
  return cudaSuccess;
}

// This is the brain dead algorithm. Just spin on an atomic until you get the
// lock.
__device__ void cudaMutexSpinLock(const cudaMutex_t mutex,
                                  unsigned int * mutexBufferHeads,
                                  const int NUM_SM)
{
  __shared__ int done;
  const bool isMasterThread = ((threadIdx.x == 0) && (threadIdx.y == 0) &&
                               (threadIdx.z == 0));
  if (isMasterThread) { done = 0; }
  __syncthreads();

  while (!done)
  {
    __syncthreads();
    if (isMasterThread)
    {
      if (atomicCAS(mutexBufferHeads + (mutex * NUM_SM), 0, 1) == 0) {
        // atomicCAS acts as a load acquire, need TF to enforce ordering
        __threadfence();
        done = 1;
      }
    }
    __syncthreads();
  }
}

__device__ void cudaMutexSpinUnlock(const cudaMutex_t mutex,
                                    unsigned int * mutexBufferHeads,
                                    const int NUM_SM)
{
  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
  {
    // atomicExch acts as a store release, need TF to enforce ordering
    __threadfence();
    atomicExch(mutexBufferHeads + (mutex * NUM_SM), 0);
  }
  __syncthreads();
}

// same algorithm but uses local TF instead because data is local
__device__ void cudaMutexSpinLockLocal(const cudaMutex_t mutex,
                                       const unsigned int smID,
                                       unsigned int * mutexBufferHeads,
                                       const int NUM_SM)
{
  __shared__ int done;
  const bool isMasterThread = ((threadIdx.x == 0) && (threadIdx.y == 0) &&
                               (threadIdx.z == 0));
  if (isMasterThread) { done = 0; }
  __syncthreads();

  while (!done)
  {
    __syncthreads();
    if (isMasterThread)
    {
      if (atomicCAS(mutexBufferHeads + ((mutex * NUM_SM) + smID), 0, 1) == 0)
      {
        // atomicCAS acts as a load acquire, need TF to enforce ordering locally
        __threadfence_block();
        done = 1;
      }
    }
    __syncthreads();
  }
}

// same algorithm but uses local TF instead because data is local
__device__ void cudaMutexSpinUnlockLocal(const cudaMutex_t mutex,
                                         const unsigned int smID,
                                         unsigned int * mutexBufferHeads,
                                         const int NUM_SM)
{
  __syncthreads();
  if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0))
  {
    // atomicExch acts as a store release, need TF to enforce ordering locally
    __threadfence_block();
    // mutex math allows us to access the appropriate per-SM spin mutex location
    atomicExch(mutexBufferHeads + ((mutex * NUM_SM) + smID), 0);
  }
  __syncthreads();
}

#endif
