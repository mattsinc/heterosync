#ifndef __HIPLOCKSMUTEXFA_CU__
#define __HIPLOCKSMUTEXFA_CU__

#include "hip/hip_runtime.h"
#include "hipLocks.h"

inline __host__ hipError_t hipMutexCreateFA(hipMutex_t * const handle,
                                            const int mutexNumber)
{
  *handle = mutexNumber;
  return hipSuccess;
}

inline __device__ void hipMutexFALock(const hipMutex_t mutex,
                                      unsigned int * mutexBufferHeads,
                                      unsigned int * mutexBufferTails,
                                      const int NUM_SM)
{
  const bool isMasterThread = (hipThreadIdx_x == 0 && hipThreadIdx_y == 0 &&
                               hipThreadIdx_z == 0);
  __shared__ unsigned int myTicketNum;
  __shared__ bool haveLock;
  const unsigned int maxTurnNum = 1000000000;

  unsigned int * const ticketNumber = mutexBufferHeads + (mutex * NUM_SM);
  volatile unsigned int * const turnNumber =
      (volatile unsigned int * )mutexBufferTails + (mutex * NUM_SM);

  __syncthreads();
  if (isMasterThread)
  {
    // load below provides ordering, no TF needed
    myTicketNum = atomicInc(ticketNumber, maxTurnNum);
    haveLock = false;
  }
  __syncthreads();
  while (!haveLock)
  {
    if (isMasterThread)
    {
      unsigned int currTicketNum = *turnNumber;

      // it's my turn, I get the lock now
      if (currTicketNum == myTicketNum) {
        // above acts as a load acquire, so need TF to enforce ordering
        __threadfence();
        haveLock = true;
      }
    }
    __syncthreads();
  }
}

inline __device__ void hipMutexFAUnlock(const hipMutex_t mutex,
                                        unsigned int * mutexBufferTails,
                                        const int NUM_SM)
{
  const bool isMasterThread = (hipThreadIdx_x == 0 && hipThreadIdx_y == 0 &&
                               hipThreadIdx_z == 0);
  const unsigned int maxTurnNum = 1000000000;
  unsigned int * const turnNumber = mutexBufferTails + (mutex * NUM_SM);

  __syncthreads();
  if (isMasterThread) {
    // atomicInc acts as a store release, need TF to enforce ordering
    __threadfence();
	/*
	  HIP currently doesn't generate the correct code for atomicInc's here,
	  so replace with an atomicAdd of 1 and assume no wraparound
	*/
    //atomicInc(turnNumber, maxTurnNum);
	atomicAdd(turnNumber, 1);
  }
  __syncthreads();
}

// same algorithm but uses per-SM lock
inline __device__ void hipMutexFALockLocal(const hipMutex_t mutex,
                                           const unsigned int smID,
                                           unsigned int * mutexBufferHeads,
                                           unsigned int * mutexBufferTails,
                                           const int NUM_SM)
{
  // local variables
  const bool isMasterThread = (hipThreadIdx_x == 0 && hipThreadIdx_y == 0 &&
                               hipThreadIdx_z == 0);
  __shared__ unsigned int myTicketNum;
  __shared__ bool haveLock;
  const unsigned int maxTurnNum = 100000000;

  unsigned int * const ticketNumber = mutexBufferHeads + ((mutex * NUM_SM) +
                                                          smID);
  volatile unsigned int * const turnNumber =
      (volatile unsigned int *)mutexBufferTails + ((mutex * NUM_SM) + smID);

  __syncthreads();
  if (isMasterThread)
  {
    myTicketNum = atomicInc(ticketNumber, maxTurnNum);
    haveLock = false;
  }
  __syncthreads();
  while (!haveLock)
  {
    if (isMasterThread)
    {
      unsigned int currTicketNum = *turnNumber;

      // it's my turn, I get the lock now
      if (currTicketNum == myTicketNum) {
        // above acts as a load acquire, so need TF to enforce ordering locally
        __threadfence_block();
        haveLock = true;
      }
    }
    __syncthreads();
  }
}

// same algorithm but uses per-SM lock
inline __device__ void hipMutexFAUnlockLocal(const hipMutex_t mutex,
                                             const unsigned int smID,
                                             unsigned int * mutexBufferTails,
                                             const int NUM_SM)
{
  const bool isMasterThread = (hipThreadIdx_x == 0 && hipThreadIdx_y == 0 &&
                               hipThreadIdx_z == 0);
  const unsigned int maxTurnNum = 100000000;

  unsigned int * const turnNumber = mutexBufferTails + ((mutex * NUM_SM) + smID);

  __syncthreads();
  if (isMasterThread) {
    // atomicInc acts as a store release, need TF to enforce ordering locally
    __threadfence_block();
	/*
	  HIP currently doesn't generate the correct code for atomicInc's here,
	  so replace with an atomicAdd of 1 and assume no wraparound
	*/
    //atomicInc(turnNumber, maxTurnNum);
	atomicAdd(turnNumber, 1);
  }
  __syncthreads();
}

#endif
