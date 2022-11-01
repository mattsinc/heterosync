#ifndef __HIPLOCKSBARRIERATOMICSRB_CU__
#define __HIPLOCKSBARRIERATOMICSRB_CU__

#include "hip/hip_runtime.h"
#include "hipLocks.h"

__device__ __forceinline__ bool ld_gbl_cg (const bool *addr)
{
  uint64_t out;
  // use GLC modifier to cache the load in L2 and bypass L1
  asm volatile (
		"flat_load_dwordx2 %0, %1 glc\n"
		"s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
		: "=v"(out) : "v"(addr): "memory"
		);
  return (bool)out;
}

inline __device__ void hipBarrierAtomicNaiveSRB(unsigned int *globalBarr,
                                                // numBarr represents the number
                                                // of WGs going to the barrier
                                                const unsigned int numBarr,
                                                int backoff,
                                                const bool isMasterThread,
                                                bool *volatile global_sense) {
  __syncthreads();
  __shared__ bool s;
  if (isMasterThread) {
    s = !ld_gbl_cg(global_sense);
    __threadfence();
    // atomicInc effectively adds 1 to atomic for each WG that's part of the
    // global barrier.
    atomicInc(globalBarr, 0x7FFFFFFF);
  }
  __syncthreads();

  while (ld_gbl_cg(global_sense) != s) {
    if (isMasterThread) {
      /*
      Once the atomic's value == numBarr, then reset the value to 0 and
      proceed because all of the WGs have reached the global barrier.
      */
      if (atomicCAS(globalBarr, numBarr, 0) == numBarr) {
        // atomicCAS acts as a load acquire, need TF to enforce ordering
        __threadfence();
        *global_sense = s;
      } else { // increase backoff to avoid repeatedly hammering global barrier
        // (capped) exponential backoff
        backoff = (((backoff << 1) + 1) & (MAX_BACKOFF - 1));
      }
    }
    __syncthreads();

    // do exponential backoff to reduce the number of times we pound the global
    // barrier
    if (ld_gbl_cg(global_sense) != s) {
      for (int i = 0; i < backoff; ++i) {
        ;
      }
      __syncthreads();
    }
  }
}

inline __device__ void hipBarrierAtomicSubSRB(unsigned int * globalBarr,
                                               // numBarr represents the number of
                                               // WGs going to the barrier
                                               const unsigned int numBarr,
                                               int backoff,
                                               const bool isMasterThread,
                                               bool * volatile sense,
                                               bool * volatile global_sense)
{
  __syncthreads();
  if (isMasterThread)
  {
    // atomicInc acts as a store release, need TF to enforce ordering
    __threadfence();
    // atomicInc effectively adds 1 to atomic for each WG that's part of the
    // global barrier.
    atomicInc(globalBarr, 0x7FFFFFFF);
  }
  __syncthreads();

  while (*global_sense != *sense)
  {
    if (isMasterThread)
    {
      /*
        For the tree barrier we expect only 1 WG from each CU to enter the
        global barrier.  Since we are assuming an equal amount of work for all
        CUs, we can use the # of WGs reaching the barrier for the compare value
        here.  Once the atomic's value == numBarr, then reset the value to 0 and
        proceed because all of the WGs have reached the global barrier.
      */
      if (atomicCAS(globalBarr, numBarr, 0) == numBarr) {
        // atomicCAS acts as a load acquire, need TF to enforce ordering
        __threadfence();
        *global_sense = *sense;
      }
      else { // increase backoff to avoid repeatedly hammering global barrier
        // (capped) exponential backoff
        backoff = (((backoff << 1) + 1) & (MAX_BACKOFF-1));
      }
    }
    __syncthreads();

    // do exponential backoff to reduce the number of times we pound the global
    // barrier
    if (*global_sense != *sense) {
      for (int i = 0; i < backoff; ++i) { ; }
      __syncthreads();
    }
  }
}

inline __device__ void hipBarrierAtomicSRB(unsigned int * barrierBuffers,
                                            // numBarr represents the number of
                                            // WGs going to the barrier
                                            const unsigned int numBarr,
                                            const bool isMasterThread,
                                            bool * volatile sense,
                                            bool * volatile global_sense)
{
  unsigned int * atomic1 = barrierBuffers;
  __shared__ int backoff;

  if (isMasterThread) {
    backoff = 1;
  }
  __syncthreads();
  hipBarrierAtomicSubSRB(atomic1, numBarr, backoff, isMasterThread, sense, global_sense);
}

inline __device__ void hipBarrierAtomicSubLocalSRB(unsigned int * perCUBarr,
                                                    const unsigned int numWGs_thisCU,
                                                    const bool isMasterThread,
                                                    bool * sense,
                                                    const int cuID)

{
  __syncthreads();
  __shared__ bool s;
  if (isMasterThread)
  {
    s = !(*sense);
    // atomicInc acts as a store release, need TF to enforce ordering locally
    __threadfence_block();
    /*
      atomicInc effectively adds 1 to atomic for each WG that's part of the
      barrier.  For the local barrier, this requires using the per-CU
      locations.
    */
    atomicInc(perCUBarr, 0x7FFFFFFF);
  }
  __syncthreads();

  while (*sense != s)
  {
    if (isMasterThread)
    {
      /*
        Once all of the WGs on this CU have incremented the value at atomic,
        then the value (for the local barrier) should be equal to the # of WGs
        on this CU.  Once that is true, then we want to reset the atomic to 0
        and proceed because all of the WGs on this CU have reached the local
        barrier.
      */
      if (atomicCAS(perCUBarr, numWGs_thisCU, 0) == numWGs_thisCU) {
        // atomicCAS acts as a load acquire, need TF to enforce ordering
        // locally
        __threadfence_block();
        *sense = s;
      }
    }
    __syncthreads();
  }
}

//Implements PerCU sense reversing barrier
inline __device__ void hipBarrierAtomicLocalSRB(unsigned int * perCUBarrierBuffers,
                                                 const unsigned int cuID,
                                                 const unsigned int numWGs_thisCU,
                                                 const bool isMasterThread,
                                                 const int MAX_BLOCKS)
{
  // each CU has MAX_BLOCKS locations in barrierBuffers, so my CU's locations
  // start at barrierBuffers[cuID*MAX_BLOCKS]
  unsigned int * atomic1 = perCUBarrierBuffers + (cuID * MAX_BLOCKS);
  bool *   sense = (bool *)(perCUBarrierBuffers + (cuID * MAX_BLOCKS) + 2);

  hipBarrierAtomicSubLocalSRB(atomic1, numWGs_thisCU, isMasterThread, sense, cuID);
}

/*
  Helper function for joining the barrier with the atomic tree barrier.
*/
__device__ void joinBarrier_helperSRB(unsigned int * barrierBuffers,
                                      unsigned int * perCUBarrierBuffers,
                                      const unsigned int numBlocksAtBarr,
                                      const int cuID,
                                      const int perCU_blockID,
                                      const int numWGs_perCU,
                                      const bool isMasterThread,
                                      const int MAX_BLOCKS) {
  bool * volatile  global_sense = (bool *)(barrierBuffers + 2);
  if (numWGs_perCU > 4) {
    bool * volatile sense = (bool *)(perCUBarrierBuffers + (cuID * MAX_BLOCKS) + 2);
    bool * volatile done = (bool *)(barrierBuffers + 1);
    *done = 0;
    __syncthreads();
    hipBarrierAtomicLocalSRB(perCUBarrierBuffers, cuID, numWGs_perCU, isMasterThread, MAX_BLOCKS);
    // only 1 WG per CU needs to do the global barrier since we synchronized
    // the WGs locally first
    if (perCU_blockID == 0) {
      hipBarrierAtomicSRB(barrierBuffers, numBlocksAtBarr, isMasterThread, sense, global_sense);  
      if(isMasterThread){
        *done = 1;
      }
      __syncthreads();
    }
    else {
      if(isMasterThread){
        while(ld_gbl_cg(done)) {;}
        __threadfence();
        while(ld_gbl_cg(global_sense) != ld_gbl_cg(sense)) {;}
      } 
      
      __syncthreads();
    }
  } else { // For low contention just call 1 level barrier
    __shared__ int backoff;
    if (isMasterThread) {
      backoff = 1;
    }
    hipBarrierAtomicNaiveSRB(barrierBuffers, gridDim.x , backoff, isMasterThread, global_sense);
  }
}

/*
  Helper function for joining the barrier with the naive atomic tree barrier 
  where all threads join the barrier.
*/
__device__ void joinBarrier_helperNaiveAllSRB(unsigned int * barrierBuffers,
					      unsigned int * perCUBarrierBuffers,
                                              const unsigned int numThreadsAtBarr,
					      const int cuID,
                                              const int MAX_BLOCKS) {
  bool * volatile global_sense = (bool *)(barrierBuffers + 2);
  bool * volatile sense = (bool *)(perCUBarrierBuffers + (cuID * MAX_BLOCKS) + 2);
  *sense = !(*global_sense);
  __syncthreads();
  // since all threads are joining, isMasterThread is "true" for all threads
  hipBarrierAtomicSRB(barrierBuffers, numThreadsAtBarr, true, sense, global_sense);
}

/*
  Helper function for joining the barrier with the naive atomic tree barrier.
*/
__device__ void joinBarrier_helperNaiveSRB(unsigned int * barrierBuffers,
                                           const unsigned int numBlocksAtBarr,
                                           const int cuID,
                                           const bool isMasterThread,
                                           const int MAX_BLOCKS) {
  bool * volatile global_sense = (bool *)(barrierBuffers + 2);
  __shared__ int backoff;
  if (isMasterThread) {
    backoff = 1;
  }
  hipBarrierAtomicNaiveSRB(barrierBuffers, numBlocksAtBarr, backoff, isMasterThread, global_sense);
}

#endif
