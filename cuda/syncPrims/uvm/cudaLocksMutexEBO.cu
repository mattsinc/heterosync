#ifndef __CUDALOCKSMUTEXEBO_CU__
#define __CUDALOCKSMUTEXEBO_CU__

#include "cudaLocks.h"

inline __host__ cudaError_t cudaMutexCreateEBO(cudaMutex_t * const handle,
                                               const int mutexNumber)
{
  *handle = mutexNumber;
  return cudaSuccess;
}

inline __device__ void cudaMutexEBOLock(const cudaMutex_t mutex,
                                        unsigned int * mutexBufferHeads,
                                        const int NUM_SM)
{
  // local variables
  __shared__ int done, iter, backoff;
  const bool isMasterThread = (threadIdx.x == 0 && threadIdx.y == 0 &&
                               threadIdx.z == 0);
  unsigned int * mutexHeadPtr = NULL;

  if (isMasterThread)
  {
    iter = 0;
    backoff = 10;
    done = 0;
    mutexHeadPtr = (mutexBufferHeads + (mutex * NUM_SM));
  }
  __syncthreads();
  while (!done)
  {
    __syncthreads();
    if (isMasterThread)
    {
      // try to acquire the lock
      if (atomicCAS(mutexHeadPtr, 0, 1) == 0) {
        // atomicCAS acts as a load acquire, need TF to enforce ordering
        __threadfence();
        done = 1;
      }
      else
      {
        // if we failed in acquiring the lock, wait for a little while before
        // trying again
        for (int j = 0; j < backoff; ++j) { ; }
        backoff += 5; // increase backoff linearly
        ++iter; // track how long we've been trying
        // if we've been waiting for a long time, wrap around and try to get the
        // lock more frequently
        if (iter > 25)
        {
          iter = 0;
          backoff = 1;
        }
      }
    }
    __syncthreads();
  }
}

inline __device__ void cudaMutexEBOUnlock(const cudaMutex_t mutex,
                                          unsigned int * mutexBufferHeads,
                                          const int NUM_SM)
{
  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    // atomicExch acts as a store release, need TF to enforce ordering
    __threadfence();
    atomicExch(mutexBufferHeads + (mutex * NUM_SM), 0); // release the lock
  }
  __syncthreads();
}

// same locking algorithm but with local scope
inline __device__ void cudaMutexEBOLockLocal(const cudaMutex_t mutex,
                                             const unsigned int smID,
                                             unsigned int * mutexBufferHeads,
                                             const int NUM_SM)
{
  // local variables
  __shared__ int done, iter, backoff;
  const bool isMasterThread = (threadIdx.x == 0 && threadIdx.y == 0 &&
                               threadIdx.z == 0);
  unsigned int * mutexHeadPtr = NULL;

  if (isMasterThread)
  {
    iter = 0;
    backoff = 10;
    done = 0;
    mutexHeadPtr = (mutexBufferHeads + ((mutex * NUM_SM) + smID));
  }
  __syncthreads();
  while (!done)
  {
    __syncthreads();
    if (isMasterThread)
    {
      // try to acquire the lock
      if (atomicCAS(mutexHeadPtr, 0, 1) == 0) {
        // atomicCAS acts as a load acquire, need TF to enforce ordering locally
        __threadfence_block();
        done = 1;
      }
      else
      {
        // if we failed in acquiring the lock, wait for a little while before
        // trying again
        for (int j = 0; j < backoff; ++j) { ; }
        backoff += 5; // increase backoff linearly
        ++iter; // track how long we've been trying
        // if we've been waiting for a long time, wrap around and try to get the
        // lock more frequently
        if (iter > 25)
        {
          iter = 0;
          backoff = 1;
        }
      }
    }
    __syncthreads();
  }
}

// same unlock algorithm but with local scope
inline __device__ void cudaMutexEBOUnlockLocal(const cudaMutex_t mutex,
                                               const unsigned int smID,
                                               unsigned int * mutexBufferHeads,
                                               const int NUM_SM)
{
  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    // atomicExch acts as a store release, need TF to enforce ordering locally
    __threadfence_block();
    atomicExch(mutexBufferHeads + ((mutex * NUM_SM) + smID), 0); // release the lock
  }
  __syncthreads();
}

#endif
