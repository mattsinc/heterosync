#ifndef __CUDALOCKSCONDITIONVARIABLE_CU__
#define __CUDALOCKSCONDITIONVARIABLE_CU__

#include "cudaLocks.h"

inline __host__ cudaError_t cudaConditionVariableCreate(cudaCondvar_t * const handle, const int condVarNumber)
{
  *handle = condVarNumber;
  return cudaSuccess;
}

inline __device__ void cudaCondvarWait(const cudaCondvar_t condvar,
                                       const cudaMutex_t mutex,
                                       const int maxBufferSize,
                                       const int arrayStride,
                                       int * conditionVariableBuffers,
                                       int * conditionVariableWaitBuffers,
                                       int * conditionVariableBufferTails,
                                       int * mutexBuffers,
                                       unsigned int * mutexBufferHeads,
                                       const int NUM_SM)
{
  const int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  const bool isMasterThread = (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0);
  int * const waitBuffer = conditionVariableWaitBuffers + arrayStride * (condvar * NUM_SM);

  // first we set ourselves as waiting.
  if (isMasterThread) { waitBuffer[blockID] = 1; }
  __syncthreads();

  /*
    now we notify any potential signalers or broadcasters that there is
    someone waiting. I'm not sure the atomic is necessary, but it makes it
    easy to increment and go back to the beginning, so it's fine for now.
  */
  int * const buffer = conditionVariableBuffers + arrayStride * (condvar * NUM_SM);
  unsigned int * const bufferTailPtr = (unsigned int *)(conditionVariableBufferTails + (condvar * NUM_SM));
  unsigned int val = atomicInc(bufferTailPtr, maxBufferSize);
  buffer[val] = blockID;

  // now we can unlock the mutex because we're all set up.
  // NOTE: This does the release for us
  cudaMutexSpinUnlock(mutex, mutexBufferHeads, NUM_SM);

  // keep checking our position in the wait buffer.
  __shared__ bool done;

  if (isMasterThread) { done = false; }
  __syncthreads();
  while (!done)
  {
    __syncthreads();
    if (isMasterThread)
    {
      if (conditionVariableWaitBuffers[blockID] == 0) { done = true; }
    }
    __syncthreads();
  }
  // NOTE: This does the acquire for us
  cudaMutexSpinLock(mutex, mutexBufferHeads, NUM_SM);
}

// Sends a broadcast to the next available sleeping block, telling it to wake
// up and proceed.
__device__ void cudaCondvarSignal(const cudaCondvar_t condvar,
                                  int maxBufferSize, int arrayStride,
                                  int * conditionVariableBuffers,
                                  int * conditionVariableWaitBuffers,
                                  int * conditionVariableBufferHeads,
                                  int * conditionVariableBufferTails,
                                  const int NUM_SM)
{
  // again, we already have control of the mutex here (or at least we better).
  __syncthreads();
  const bool isMasterThread = (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0);
  __shared__ bool empty;
  if (isMasterThread) { empty = (conditionVariableBufferHeads[(condvar * NUM_SM)] == conditionVariableBufferTails[(condvar * NUM_SM)]); }
  __syncthreads();

  if (empty) { return; } // there's nothing to do.

  // in this case, there is at least one block waiting to be notified. find
  // the blockID and set its wait buffer index to 0, telling it to try and
  // lock the mutex.
  if (isMasterThread)
  {
    int * const buffer = conditionVariableBuffers + arrayStride * (condvar * NUM_SM);
    unsigned int * const waitBuffer = (unsigned int *)(conditionVariableWaitBuffers + arrayStride * (condvar * NUM_SM));
    // again, we have a lock on the condvar, so this atomic is probably
    // unnecessary.
    unsigned int val = atomicInc((unsigned int *)(conditionVariableBufferHeads + (condvar * NUM_SM)), maxBufferSize);
    int blockID = buffer[val];
    waitBuffer[blockID] = 0;
  }
  __syncthreads();
}

// Sends a broadcast to all sleeping blocks telling them wake up and proceed.
// Called with an entire block.
inline __device__ void cudaCondvarBroadcast(const cudaCondvar_t condvar,
                                            int maxBufferSize, int arrayStride,
                                            int * conditionVariableBuffers,
                                            int * conditionVariableWaitBuffers,
                                            int * conditionVariableBufferHeads,
                                            int * conditionVariableBufferTails,
                                            const int NUM_SM)
{
  // already have the lock if we're here.
  __shared__ int startIndex;
  __shared__ int endIndex;
  const bool isMasterThread = ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0));

  // get the indices for the blocks that are waiting.
  if (isMasterThread)
  {
    startIndex = conditionVariableBufferHeads[(condvar * NUM_SM)];
    endIndex = conditionVariableBufferTails[(condvar * NUM_SM)];
    if (endIndex < startIndex) { endIndex += maxBufferSize; }
    conditionVariableBufferHeads[(condvar * NUM_SM)] = conditionVariableBufferTails[(condvar * NUM_SM)] = 0;
  }
  __syncthreads();

  // since we're calling this using a block, we can intelligently go through
  // and tell everyone that they can try and lock the mutex now.
  const int threadID = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
  int * const buffer      = conditionVariableBuffers + (condvar * NUM_SM) * arrayStride;
  int * const waitBuffer  = conditionVariableWaitBuffers + (condvar * NUM_SM) * arrayStride;
  for (int i = startIndex + threadID; i < endIndex; ++i)
  {
    waitBuffer[buffer[i % maxBufferSize]] = 0;
  }
  __syncthreads();
}

// same algorithm but with per-SM synchronization
inline __device__ void cudaCondvarWaitLocal(const cudaCondvar_t condvar,
                                            const cudaMutex_t mutex,
                                            const unsigned int smID,
                                            const int maxBufferSize,
                                            const int arrayStride,
                                            int * conditionVariableBuffers,
                                            int * conditionVariableWaitBuffers,
                                            int * conditionVariableBufferTails,
                                            int * mutexBuffers,
                                            unsigned int * mutexBufferHeads,
                                            const int NUM_SM)
{
  const int blockID = blockIdx.y * gridDim.x + blockIdx.x;
  const bool isMasterThread = (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0);
  int * const waitBuffer = conditionVariableWaitBuffers + arrayStride * ((condvar * NUM_SM) + smID);

  // first we set ourselves as waiting.
  if (isMasterThread) { waitBuffer[blockID] = 1; }
  __syncthreads();

  // now we notify any potential signalers or broadcasters that there is
  // someone waiting. i'm not sure the atomic is necessary, but it makes it
  // easy to increment and go back to the beginning, so it's fine for now.
  int * const buffer = conditionVariableBuffers + arrayStride * ((condvar * NUM_SM) + smID);
  unsigned int * const bufferTailPtr = (unsigned int *)(conditionVariableBufferTails + ((condvar * NUM_SM) + smID));
  unsigned int val = atomicInc(bufferTailPtr, maxBufferSize);
  buffer[val] = blockID;

  // now we can unlock the mutex because we're all set up.
  // NOTE: This does the release for us
  cudaMutexSpinUnlockLocal(mutex, smID, mutexBufferHeads, NUM_SM);

  // keep checking our position in the wait buffer.
  __shared__ bool done;

  if (isMasterThread) { done = false; }
  __syncthreads();
  while (!done)
  {
    __syncthreads();
    if (isMasterThread)
    {
      if (conditionVariableWaitBuffers[blockID] == 0) { done = true; }
    }
    __syncthreads();
  }
  // NOTE: This does the acquire for us
  cudaMutexSpinLockLocal(mutex, smID, mutexBufferHeads, NUM_SM);
}

// same algorithm but with per-SM synchronization
// Sends a broadcast to the next available sleeping block, telling it to wake
// up and proceed.
__device__ void cudaCondvarSignalLocal(const cudaCondvar_t condvar,
                                       const unsigned int smID,
                                       int maxBufferSize, int arrayStride,
                                       int * conditionVariableBuffers,
                                       int * conditionVariableWaitBuffers,
                                       int * conditionVariableBufferHeads,
                                       int * conditionVariableBufferTails,
                                       const int NUM_SM)
{
  // again, we already have control of the mutex here (or at least we better).
  __syncthreads();
  const bool isMasterThread = (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0);
  __shared__ bool empty;
  if (isMasterThread) { empty = (conditionVariableBufferHeads[((condvar * NUM_SM) + smID)] == conditionVariableBufferTails[((condvar * NUM_SM) + smID)]); }
  __syncthreads();

  if (empty) { return; } // there's nothing to do.

  // in this case, there is at least one block waiting to be notified. find
  // the blockID and set its wait buffer index to 0, telling it to try and
  // lock the mutex.
  if (isMasterThread)
  {
    int * const buffer = conditionVariableBuffers + arrayStride * ((condvar * NUM_SM) + smID);
    unsigned int * const waitBuffer = (unsigned int *)(conditionVariableWaitBuffers + arrayStride * ((condvar * NUM_SM) + smID));
    // again, we have a lock on the condvar, so this atomic is probably
    // unnecessary.
    unsigned int val = atomicInc((unsigned int *)(conditionVariableBufferHeads + ((condvar * NUM_SM) + smID)), maxBufferSize);
    int blockID = buffer[val];
    waitBuffer[blockID] = 0;
  }
  __syncthreads();
}

// same algorithm but with per-SM synchronization
// Sends a broadcast to all sleeping blocks telling them wake up and proceed.
// Called with an entire block.
inline __device__ void cudaCondvarBroadcastLocal(const cudaCondvar_t condvar,
                                                 const unsigned int smID,
                                                 int maxBufferSize,
                                                 int arrayStride,
                                                 int * conditionVariableBuffers,
                                                 int * conditionVariableWaitBuffers,
                                                 int * conditionVariableBufferHeads,
                                                 int * conditionVariableBufferTails,
                                                 const int NUM_SM)
{
  // already have the lock if we're here.
  __shared__ int startIndex;
  __shared__ int endIndex;
  const bool isMasterThread = ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0));

  // get the indices for the blocks that are waiting.
  if (isMasterThread)
  {
    startIndex = conditionVariableBufferHeads[((condvar * NUM_SM) + smID)];
    endIndex = conditionVariableBufferTails[((condvar * NUM_SM) + smID)];
    if (endIndex < startIndex) { endIndex += maxBufferSize; }
    conditionVariableBufferHeads[((condvar * NUM_SM) + smID)] = conditionVariableBufferTails[((condvar * NUM_SM) + smID)] = 0;
  }
  __syncthreads();

  // since we're calling this using a block, we can intelligently go through
  // and tell everyone that they can try and lock the mutex now.
  const int threadID = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
  int * const buffer      = conditionVariableBuffers + ((condvar * NUM_SM) + smID) * arrayStride;
  int * const waitBuffer  = conditionVariableWaitBuffers + ((condvar * NUM_SM) + smID) * arrayStride;
  for (int i = startIndex + threadID; i < endIndex; ++i)
  {
    waitBuffer[buffer[i % maxBufferSize]] = 0;
  }
  __syncthreads();
}

#endif
