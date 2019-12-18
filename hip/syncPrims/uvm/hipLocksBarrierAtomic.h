#ifndef __HIPLOCKSBARRIERATOMIC_CU__
#define __HIPLOCKSBARRIERATOMIC_CU__

#include "hip/hip_runtime.h"
#include "hipLocks.h"

inline __device__ void hipBarrierAtomicSub(unsigned int * globalBarr,
                                            int * done,
                                            // numBarr represents the number of
                                            // TBs going to the barrier
                                            const unsigned int numBarr,
                                            int backoff,
                                            const bool isMasterThread)
{
  __syncthreads();
  if (isMasterThread)
  {
    *done = 0;

    // atomicInc acts as a store release, need TF to enforce ordering
    __threadfence();
    // atomicInc effectively adds 1 to atomic for each TB that's part of the
    // global barrier.
    atomicInc(globalBarr, 0x7FFFFFFF);
  }
  __syncthreads();

  while (!*done)
  {
    if (isMasterThread)
    {
      /*
        For the tree barrier we expect only 1 TB from each SM to enter the
        global barrier.  Since we are assuming an equal amount of work for all
        SMs, we can use the # of TBs reaching the barrier for the compare value
        here.  Once the atomic's value == numBarr, then reset the value to 0 and
        proceed because all of the TBs have reached the global barrier.
      */
      if (atomicCAS(globalBarr, numBarr, 0) == 0) {
        // atomicCAS acts as a load acquire, need TF to enforce ordering
        __threadfence();
        *done = 1;
      }
      else { // increase backoff to avoid repeatedly hammering global barrier
        // (capped) exponential backoff
        backoff = (((backoff << 1) + 1) & (MAX_BACKOFF-1));
      }
    }
    __syncthreads();

    // do exponential backoff to reduce the number of times we pound the global
    // barrier
    if (!*done) {
      for (int i = 0; i < backoff; ++i) { ; }
      __syncthreads();
    }
  }
}

inline __device__ void hipBarrierAtomic(unsigned int * barrierBuffers,
                                         // numBarr represents the number of
                                         // TBs going to the barrier
                                         const unsigned int numBarr,
                                         const bool isMasterThread)
{
  unsigned int * atomic1 = barrierBuffers;
  unsigned int * atomic2 = atomic1 + 1;
  __shared__ int done1, done2;
  __shared__ int backoff;

  if (isMasterThread) {
    backoff = 1;
  }
  __syncthreads();

  hipBarrierAtomicSub(atomic1, &done1, numBarr, backoff, isMasterThread);
  // second barrier is necessary to provide a facesimile for a sense-reversing
  // barrier
  hipBarrierAtomicSub(atomic2, &done2, numBarr, backoff, isMasterThread);
}

// does local barrier amongst all of the TBs on an SM
inline __device__ void hipBarrierAtomicSubLocal(unsigned int * perSMBarr,
                                                 int * done,
                                                 const unsigned int numTBs_thisSM,
                                                 const bool isMasterThread)
{
  __syncthreads();
  if (isMasterThread)
  {
    *done = 0;
    // atomicInc acts as a store release, need TF to enforce ordering locally
    __threadfence_block();
    /*
      atomicInc effectively adds 1 to atomic for each TB that's part of the
      barrier.  For the local barrier, this requires using the per-CU
      locations.
    */
    atomicInc(perSMBarr, 0x7FFFFFFF);
  }
  __syncthreads();

  while (!*done)
  {
    if (isMasterThread)
    {
      /*
        Once all of the TBs on this SM have incremented the value at atomic,
        then the value (for the local barrier) should be equal to the # of TBs
        on this SM.  Once that is true, then we want to reset the atomic to 0
        and proceed because all of the TBs on this SM have reached the local
        barrier.
      */
      if (atomicCAS(perSMBarr, numTBs_thisSM, 0) == 0) {
        // atomicCAS acts as a load acquire, need TF to enforce ordering
        // locally
        __threadfence_block();
        *done = 1;
      }
    }
    __syncthreads();
  }
}

// does local barrier amongst all of the TBs on an SM
inline __device__ void hipBarrierAtomicLocal(unsigned int * perSMBarrierBuffers,
                                              const unsigned int smID,
                                              const unsigned int numTBs_thisSM,
                                              const bool isMasterThread,
                                              const int MAX_BLOCKS)
{
  // each SM has MAX_BLOCKS locations in barrierBuffers, so my SM's locations
  // start at barrierBuffers[smID*MAX_BLOCKS]
  unsigned int * atomic1 = perSMBarrierBuffers + (smID * MAX_BLOCKS);
  unsigned int * atomic2 = atomic1 + 1;
  __shared__ int done1, done2;

  hipBarrierAtomicSubLocal(atomic1, &done1, numTBs_thisSM, isMasterThread);
  // second barrier is necessary to approproximate a sense-reversing barrier
  hipBarrierAtomicSubLocal(atomic2, &done2, numTBs_thisSM, isMasterThread);
}

/*
  Helper function for joining the barrier with the atomic tree barrier.
*/
__device__ void joinBarrier_helper(unsigned int * barrierBuffers,
                                   unsigned int * perSMBarrierBuffers,
                                   const unsigned int numBlocksAtBarr,
                                   const int smID,
                                   const int perSM_blockID,
                                   const int numTBs_perSM,
                                   const bool isMasterThread,
                                   const int MAX_BLOCKS) {
  if (numTBs_perSM > 1) {
    hipBarrierAtomicLocal(perSMBarrierBuffers, smID, numTBs_perSM,
                           isMasterThread, MAX_BLOCKS);

    // only 1 TB per SM needs to do the global barrier since we synchronized
    // the TBs locally first
    if (perSM_blockID == 0) {
      hipBarrierAtomic(barrierBuffers, numBlocksAtBarr, isMasterThread);
    }

    // all TBs on this SM do a local barrier to ensure global barrier is
    // reached
    hipBarrierAtomicLocal(perSMBarrierBuffers, smID, numTBs_perSM,
                           isMasterThread, MAX_BLOCKS);
  } else { // if only 1 TB on the SM, no need for the local barriers
    hipBarrierAtomic(barrierBuffers, numBlocksAtBarr, isMasterThread);
  }
}

#endif
