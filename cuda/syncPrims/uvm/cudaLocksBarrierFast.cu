#ifndef __CUDALOCKSBARRIERFAST_CU__
#define __CUDALOCKSBARRIERFAST_CU__

#include "cudaLocks.h"

/*
  Helper function to set the passed in inVars flag to 1 (signifies that this TB
  has joined the barrier).
 */
inline __device__ void setMyInFlag(unsigned int * inVars,
                                   const unsigned int threadID,
                                   const unsigned int blockID) {
  if (threadID == 0)
  {
    // atomicExch acts as a store release, need TF to enforce ordering
    __threadfence();
    atomicExch((unsigned int *)(inVars + blockID), 1);
  }
  __syncthreads();
}

/*
  Helper function for the main TB of this group to spin, checking to see if
  all other TBs joining this barrier have joined or not.
 */
inline __device__ void spinOnInFlags(unsigned int * inVars,
                                     const int threadID,
                                     const int numThreads,
                                     const int numBlocks) {
  // local variables
  int done3 = 1;

  // "main" TB loops, checking if everyone else has joined the barrier.
  do
  {
    done3 = 1;

    /*
      Each thread in the main TB accesses a subset of the blocks, checking
      if they have joined the barrier yet or not.
    */
    for (int i = threadID; i < numBlocks; i += numThreads)
    {
      if (reinterpret_cast<volatile int * >(inVars)[i] != 1) {
        // acts as a load acquire, need TF to enforce ordering
        __threadfence();

        done3 = 0;
        // if one of them isn't ready, don't bother checking the others (just
        // increases traffic)
        break;
      }
    }
  } while (!done3);
  /*
    When all the necessary TBs have joined the barrier, the threads will
    reconverge here -- this avoids unnecessary atomic accesses for threads
    whose assigned TBs have already joined the barrier.
  */
  __syncthreads();
}

/*
  Helper function for the main TB of this group to spin, checking to see if
  all other TBs joining this barrier have joined or not.
*/
inline __device__ void spinOnInFlags_local(unsigned int * inVars,
                                           const int threadID,
                                           const int numThreads,
                                           const int numBlocks) {
  // local variables
  int done3 = 1;

  // "main" TB loops, checking if everyone else has joined the barrier.
  do
  {
    done3 = 1;

    /*
      Each thread in the main TB accesses a subset of the blocks, checking
      if they have joined the barrier yet or not.
    */
    for (int i = threadID; i < numBlocks; i += numThreads)
    {
      if (reinterpret_cast<volatile int * >(inVars)[i] != 1) {
        // acts as a load acquire, need TF to enforce ordering locally
        __threadfence_block();

        done3 = 0;
        // if one of them isn't ready, don't bother checking the others (just
        // increases traffic)
        break;
      }
    }
  } while (!done3);
  /*
    When all the necessary TBs have joined the barrier, the threads will
    reconverge here -- this avoids unnecessary atomic accesses for threads
    whose assigned TBs have already joined the barrier.
  */
  __syncthreads();
}

/*
  Helper function for main TB to set the outVars flags for all TBs at this
  barrier to notify them that everyone has joined the barrier and they can
  proceed.
*/
inline __device__ void setOutFlags(unsigned int * inVars,
                                   unsigned int * outVars,
                                   const int threadID,
                                   const int numThreads,
                                   const int numBlocks) {
  for (int i = threadID; i < numBlocks; i += numThreads)
  {
    reinterpret_cast<volatile int * >(inVars)[i] = 0;
    reinterpret_cast<volatile int * >(outVars)[i] = 1;
  }
  __syncthreads();
  // outVars acts as a store release, need TF to enforce ordering
  __threadfence();
}

/*
  Helper function for main TB to set the outVars flags for all TBs at this
  barrier to notify them that everyone has joined the barrier and they can
  proceed.
*/
inline __device__ void setOutFlags_local(unsigned int * inVars,
                                         unsigned int * outVars,
                                         const int threadID,
                                         const int numThreads,
                                         const int numBlocks) {
  for (int i = threadID; i < numBlocks; i += numThreads)
  {
    reinterpret_cast<volatile int * >(inVars)[i] = 0;
    reinterpret_cast<volatile int * >(outVars)[i] = 1;
  }
  __syncthreads();
  // outVars acts as a store release, need TF to enforce ordering locally
  __threadfence_block();
}

/*
  Helper function for each TB to spin waiting for its outVars flag to be set
  by the main TB.  When it is set, then this TB can safely exit the barrier.
*/
inline __device__ void spinOnMyOutFlag(unsigned int * inVars,
                                       unsigned int * outVars,
                                       const int blockID,
                                       const int threadID) {
  if (threadID == 0)
  {
    while (reinterpret_cast<volatile int * >(outVars)[blockID] != 1) { ; }

    inVars[blockID] = outVars[blockID] = 0;
    // these stores act as a store release, need TF to enforce ordering
    __threadfence();
  }
  __syncthreads();
}

/*
  Helper function for each TB to spin waiting for its outVars flag to be set
  by the main TB.  When it is set, then this TB can safely exit the barrier.
*/
inline __device__ void spinOnMyOutFlag_local(unsigned int * inVars,
                                             unsigned int * outVars,
                                             const int blockID,
                                             const int threadID) {
  if (threadID == 0)
  {
    while (reinterpret_cast<volatile int * >(outVars)[blockID] != 1) { ; }

    inVars[blockID] = outVars[blockID] = 0;
    // these stores act as a store release, need TF to enforce ordering locally
    __threadfence_block();
  }
  __syncthreads();
}

__device__ void cudaBarrier(unsigned int * barrierBuffers,
                            const int arrayStride,
                            const unsigned int numBlocksAtBarr)
{
  // local variables
  const int threadID = threadIdx.x;
  const int blockID = blockIdx.x;
  const int numThreads = blockDim.x;
  // ** NOTE: setting numBlocks like this only works if the first TB on
  // each SM joins the global barrier
  const int numBlocks = numBlocksAtBarr;
  unsigned int * const inVars  = barrierBuffers;
  unsigned int * const outVars = barrierBuffers + arrayStride;

  /*
    Thread 0 from each TB sets its 'private' flag in the in array to 1 to
    signify that it has joined the barrier.
  */
  setMyInFlag(inVars, threadID, blockID);

  // TB 0 is the "main" TB for the global barrier
  if (blockID == 0)
  {
    // "main" TB loops, checking if everyone else has joined the barrier.
    spinOnInFlags(inVars, threadID, numThreads, numBlocks);

    /*
      Once all the TBs arrive at the barrier, the main TB resets them to
      notify everyone else that they can move forward beyond the barrier --
      again each thread in the main TB takes a subset of the necessary TBs
      and sets their in flag to 0 and out flag to 1.
    */
    setOutFlags(inVars, outVars, threadID, numThreads, numBlocks);
  }

  /*
    All TBs (including the main one) spin, checking to see if the main one
    set their out location yet -- if it did, then they can move ahead
    because the barrier is done.
  */
  spinOnMyOutFlag(inVars, outVars, blockID, threadID);
}

// same algorithm but per-SM synchronization
__device__ void cudaBarrierLocal(// for global barrier
                                 unsigned int * barrierBuffers,
                                 const unsigned int numBlocksAtBarr,
                                 const int arrayStride,
                                 // for local barrier
                                 unsigned int * perSMBarrierBuffers,
                                 const unsigned int smID,
                                 const unsigned int numTBs_perSM,
                                 const unsigned int perSM_blockID,
                                 const bool isLocalGlobalBarr,
                                 const int MAX_BLOCKS)
{
  // local variables
  const int threadID = threadIdx.x;
  const int numThreads = blockDim.x;
  const int numBlocks = numTBs_perSM;
  /*
    Each SM has MAX_BLOCKS*2 locations in perSMBarrierBuffers, so my SM's
    inVars locations start at perSMBarrierBuffers[smID*2*MAX_BLOCKS] and my
    SM's outVars locations start at
    perSMBarrierBuffers[smID*2*MAX_BLOCKS + MAX_BLOCKS].
  */
  unsigned int * const inVars  = perSMBarrierBuffers + (MAX_BLOCKS * smID * 2);
  unsigned int * const outVars = perSMBarrierBuffers + ((MAX_BLOCKS * smID * 2) + MAX_BLOCKS);

  /*
    Thread 0 from each TB sets its 'private' flag in the in array to 1 to
    signify that it has joined the barrier.
  */
  setMyInFlag(inVars, threadID, perSM_blockID);

  // first TB on this SM is the "main" TB for the local barrier
  if (perSM_blockID == 0)
  {
    // "main" TB loops, checking if everyone else has joined the barrier.
    spinOnInFlags_local(inVars, threadID, numThreads, numBlocks);

    /*
      If we are calling the global tree barrier from within the local tree
      barrier, call it here.  Now that all of the TBs on this SM have joined
      the local barrier, TB 0 on this SM joins the global barrier.
    */
    if (isLocalGlobalBarr) {
      cudaBarrier(barrierBuffers, arrayStride, numBlocksAtBarr);
    }

    /*
      Once all the TBs arrive at the barrier, the main TB resets their inVar
      and sets their outVar to notify everyone else that they can move
      forward beyond the barrier -- each thread in the main TB takes a subset
      of the necessary TBs and sets their in flag to 0 and out flag to 1.
    */
    setOutFlags_local(inVars, outVars, threadID, numThreads, numBlocks);
  }

  /*
    All TBs (including the main one) spin, checking to see if the main TB
    set their out location yet -- if it did, then they can move ahead
    because the barrier is done.
  */
  spinOnMyOutFlag_local(inVars, outVars, perSM_blockID, threadID);
}

/*
  Decentralized tree barrier that has 1 TB per SM join the global decentralized
  barrier in the middle, then sets the out flags of the others on this SM to 1
  after returning.  This avoids the need for a second local barrier after the
  global barrier.
*/
__device__ void cudaBarrierLocalGlobal(// for global barrier
                                       unsigned int * barrierBuffers,
                                       const unsigned int numBlocksAtBarr,
                                       const int arrayStride,
                                       // for local barrier
                                       unsigned int * perSMBarrierBuffers,
                                       const unsigned int smID,
                                       const unsigned int numTBs_perSM,
                                       const unsigned int perSM_blockID,
                                       const int MAX_BLOCKS)
{
  // will call global barrier within it
  cudaBarrierLocal(barrierBuffers, numBlocksAtBarr, arrayStride,
                   perSMBarrierBuffers, smID, numTBs_perSM, perSM_blockID,
                   true, MAX_BLOCKS);
}

/*
  Helper function for joining the barrier with the 'lock-free' tree barrier.
*/
__device__ void joinLFBarrier_helper(unsigned int * barrierBuffers,
                                     unsigned int * perSMBarrierBuffers,
                                     const unsigned int numBlocksAtBarr,
                                     const int smID,
                                     const int perSM_blockID,
                                     const int numTBs_perSM,
                                     const int arrayStride,
                                     const int MAX_BLOCKS) {
  if (numTBs_perSM > 1) {
    cudaBarrierLocalGlobal(barrierBuffers, numBlocksAtBarr, arrayStride,
                           perSMBarrierBuffers, smID, numTBs_perSM,
                           perSM_blockID, MAX_BLOCKS);
  } else { // if only 1 TB on the SM, no need for the local barriers
    cudaBarrier(barrierBuffers, arrayStride, numBlocksAtBarr);
  }
}

#endif
