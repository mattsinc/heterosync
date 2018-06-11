#ifndef __CUDALOCKMUTEXSLEEP_H__
#define __CUDALOCKMUTEXSLEEP_H__

#include "cudaLocks.h"

inline __host__ cudaError_t cudaMutexCreateSleep(cudaMutex_t * const handle, const int mutexNumber)
{
  *handle = mutexNumber;
  return cudaSuccess;
}

/*
  Instead of constantly pounding an atomic to try and lock the mutex, we simply
  put ourselves into a ring buffer. Then we check our location in the ring 
  buffer to see if it's been set to 1 -- when it has, it is our turn.  When
  we're done, unset our location and set the next location to 1.

  locks the mutex. must be called by the entire block.
*/
__device__ unsigned int cudaMutexSleepLock(const cudaMutex_t mutex,
                                           int * mutexBuffers,
                                           unsigned int * mutexBufferTails,
                                           const int maxRingBufferSize,
                                           const int arrayStride,
                                           const int NUM_SM)
{
  __syncthreads();

  // local variables
  const bool isMasterThread = (threadIdx.x == 0 && threadIdx.y == 0 &&
                               threadIdx.z == 0);

  unsigned int * const ringBufferTailPtr = mutexBufferTails + (mutex * NUM_SM);
  // since this just assigns a pointer, should be ok even though it's volatile
  volatile int * const ringBuffer = (volatile int *)mutexBuffers + (mutex * NUM_SM) * arrayStride;

  __shared__ unsigned int myRingBufferLoc;
  __shared__ bool haveLock;

  // this is a fire-and-forget atomic.
  if (isMasterThread)
  {
    /*
      Use a reprogrammed atomicAnd to get the same functionality as atomicInc
      but without the store release semantics -- the atomicXor below determines
      the happens-before ordering here.
    */
    myRingBufferLoc = atomicAnd(ringBufferTailPtr, maxRingBufferSize);

    haveLock = false; // initially we don't have the lock
  }
  __syncthreads();

  //  Two possibilities
  //    Mutex is unlocked
  //    Mutex is locked
  while (!haveLock)
  {
    __syncthreads();
    if (isMasterThread)
    {
      // spin waiting for our location in the ring buffer to == 1.
      /*
        Use an acquire instead of a read to invalidate the appropriate data
        we're having a problem with the function call atomics, so use a
        reprogrammed atomicXor (reads value + acquire semantics) instead.
      */
      if (atomicXor((unsigned int *)(ringBuffer + myRingBufferLoc), 0) == 1)
      {
        // When our location in the ring buffer == 1, we have the lock
        haveLock = true;
      }
    }
    __syncthreads();
  }

  return myRingBufferLoc;
}

// to unlock, simply increment the ring buffer's head pointer.
__device__ void cudaMutexSleepUnlock(const cudaMutex_t mutex,
                                     int * mutexBuffers,
                                     unsigned int myBufferLoc,
                                     const int maxRingBufferSize,
                                     const int arrayStride,
                                     const int NUM_SM)
{
  __syncthreads();

  const bool isMasterThread = (threadIdx.x == 0 && threadIdx.y == 0 &&
                               threadIdx.z == 0);
  int * ringBuffer = (int * )mutexBuffers + (mutex * NUM_SM) * arrayStride;
  // next location is 0 if we're the last location in the buffer (wraparound)
  const unsigned int nextBufferLoc = ((myBufferLoc >= maxRingBufferSize) ? 0 :
                                      myBufferLoc + 1);

  if (isMasterThread)
  {
    /*
      Now that we have the lock, set our ring buffer location to -1 to
      prevent another TB from accidentally thinking its turn is up later on
      (subtracting 2 sets the value to -1 without doing a store release).

      Although we have the lock at this point, this still needs to be an
      atomic access to ensure that the other threads (which are still
      reading the current head pointer) are not reading stale data
    */
    atomicSub((int *)(ringBuffer + myBufferLoc), 2);

    // set the next location in the ring buffer to 1 so that next TB in line
    // can get the lock now
    atomicExch((int *)ringBuffer + nextBufferLoc, 1);
  }
  __syncthreads();
}

// same algorithm but uses per-SM lock
__device__ unsigned int cudaMutexSleepLockLocal(const cudaMutex_t mutex,
                                                const unsigned int smID,
                                                int * mutexBuffers,
                                                unsigned int * mutexBufferTails,
                                                const int maxRingBufferSize,
                                                const int arrayStride,
                                                const int NUM_SM)
{
  __syncthreads();

  // local variables
  const bool isMasterThread = (threadIdx.x == 0 && threadIdx.y == 0 &&
                               threadIdx.z == 0);
  unsigned int * const ringBufferTailPtr = mutexBufferTails + ((mutex * NUM_SM) +
                                                               smID);
  // since this just assigns a pointer, should be ok even though it's volatile
  volatile int * const ringBuffer = (volatile int * )mutexBuffers +
                                    ((mutex * NUM_SM) + smID) * arrayStride;

  __shared__ unsigned int myRingBufferLoc;
  __shared__ bool haveLock;

  // this is a fire-and-forget atomic.
  if (isMasterThread)
  {
    /*
      Use a reprogrammed atomicAnd to get the same functionality as atomicInc
      but without the store release semantics -- the atomicXor below determines
      the happens-before ordering here.
    */
    myRingBufferLoc = atomicAnd(ringBufferTailPtr, maxRingBufferSize);

    haveLock = false; // initially we don't have the lock
  }
  __syncthreads();

  //  Two possibilities
  //    Mutex is unlocked
  //    Mutex is locked
  while (!haveLock)
  {
    __syncthreads();
    if (isMasterThread)
    {
      // spin waiting for our location in the ring buffer to == 1.
      /*
        Use an acquire instead of a read to invalidate the appropriate data.
      */
      if (atomicXor((unsigned int *)(ringBuffer + myRingBufferLoc), 0) == 1)
      {
        // When our location in the ring buffer == 1, we have the lock
        haveLock = true;
      }
    }
    __syncthreads();
  }

  return myRingBufferLoc;
}

// to unlock, simply increment the ring buffer's head pointer -- same algorithm
// but uses per-SM lock.
__device__ void cudaMutexSleepUnlockLocal(const cudaMutex_t mutex,
                                          const unsigned int smID,
                                          int * mutexBuffers,
                                          unsigned int myBufferLoc,
                                          const int maxRingBufferSize,
                                          const int arrayStride,
                                          const int NUM_SM)
{
  __syncthreads();

  const bool isMasterThread = (threadIdx.x == 0 && threadIdx.y == 0 &&
                               threadIdx.z == 0);
  int * ringBuffer = (int * )mutexBuffers + ((mutex * NUM_SM) + smID) *
                     arrayStride;
  // next location is 0 if we're the last location in the buffer (wraparound)
  const unsigned int nextBufferLoc = ((myBufferLoc >= maxRingBufferSize) ? 0 :
                                      myBufferLoc + 1);

  if (isMasterThread)
  {
    /*
      Now that we have the lock, set our ring buffer location to -1 to
      prevent another TB from accidentally thinking its turn is up later on
      (subtracting 2 sets the value to -1 without doing a store release).

      Although we have the lock at this point, this still needs to be an
      atomic access to ensure that the other threads (which are still
      reading the current head pointer) are not reading stale data.
    */
    atomicSub((unsigned int *)(ringBuffer + myBufferLoc), 2);
    // set the next location in the ring buffer to 1 so that next TB in line
    // can get the lock now
    atomicExch((int *)ringBuffer + nextBufferLoc, 1);
  }
  __syncthreads();
}

#endif
