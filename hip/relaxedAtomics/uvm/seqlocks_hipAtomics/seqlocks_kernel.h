#ifndef SEQLOCKS_KERNEL_H_
#define SEQLOCKS_KERNEL_H_

#include "hip/hip_runtime.h"
#include "sleepHelper.h"

#define WAVE_SIZE 32
#define HALF_WAVE_SIZE (WAVE_SIZE >> 1)
#define NUM_ITERS 10

/*
  max exponential backoff value (need to make this desired
  power of 2 * 2 because we use bitwise ANDs of MAX_BACKOFF-1 to
  do the wraparound.
*/
#define MAX_BACKOFF 128

/*
  seqlock readers try to read the current data values, and retry if the sequence
  access numbers do not match -- because this signifies that a writer updated
  the data values in between the readers' accesses of the sequence access number.
*/
inline __device__ void reader_strong(unsigned int * seqlock,
                                     int * dataArr0,
                                     int * dataArr1,
                                     int * outArr,
                                     const unsigned int threadID,
                                     const unsigned int seqlockLoc,
                                     const unsigned int dataLoc,
                                     const bool isMasterThread) {
  // local variables
  int r1 = 0, r2 = 0;
  unsigned int seq0 = 0, seq1 = 0;

  do {
    // atomic load, need acquire semantics
    if (isMasterThread) {
      seq0 = __hip_atomic_load(&(seqlock[seqlockLoc]), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
    }
    __syncthreads();

    // stores in between seqlocks are relaxed
    /*r1 = */__hip_atomic_store(&(dataArr0[dataLoc]), 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    /*r2 = */__hip_atomic_store(&(dataArr1[dataLoc]), 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

    // need release semantics
    if (isMasterThread) {
      seq1 = __hip_atomic_load(&(seqlock[seqlockLoc]), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
    }
    __syncthreads();
  } while ((seq0 != seq1) || (seq0 & 1));

  // update my threads location in the out array with data array values
  // just use the combination of the data array values
  outArr[threadID] = r1 + r2;
}

/*
  seqlock writers attempt to update the current data values.  They retry if
  another writer is writing (seq0 is odd) or they are unable to obtain the
  "lock" on seqlock (CAS).
*/
inline __device__ void writer_strong(unsigned int * seqlock,
                                     int * dataArr0,
                                     int * dataArr1,
                                     const unsigned int seqlockLoc,
                                     const unsigned int dataLoc,
                                     const bool isMasterThread) {
  // local variables
  volatile int backoff = 1;
  unsigned int seq0 = 0, seq0_new = 0;

  if (isMasterThread) {
    // need acquire semantics for this load per SeqLocks paper
    seq0 = __hip_atomic_load(&(seqlock[seqlockLoc]), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);

    // spin until no readers or other writers
    while ((seq0 & 1) ||
	   // use CAS weak because if we fail we don't need ordering --
	   // but if we succeed we want ordering
	   __hip_atomic_compare_exchange_weak(&(seqlock[seqlockLoc]), &seq0_new, seq0+1, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != seq0) {
           //(atomicCAS(&(seqlock[seqlockLoc]), seq0, seq0+1) != seq0)) {
      // if we failed, wait for a little while before trying again
      sleepFunc(backoff);
      // backoff so don't keep hammering seqlock
      backoff = ((backoff << 1) + 1) & (MAX_BACKOFF-1);

      // reload seq0 to see any changes the other writers have made
      // use acquire semantics
      seq0 = __hip_atomic_load(&(seqlock[seqlockLoc]), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
    }
  }
  __syncthreads();
  // need fence here to handle the compare_exchange_weak semantics -- if it succeeded
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

  /*
    Write the current sequence number, value doesn't really matter.
    These accesses can be overlapped, so make first one relaxed and 
    second one release.
  */
  __hip_atomic_store(&(dataArr0[dataLoc]), seq0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  __hip_atomic_store(&(dataArr1[dataLoc]), seq0, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);

  if (isMasterThread) {
    __hip_atomic_store(&(seqlock[seqlockLoc]), seq0 + 2, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  __syncthreads();
}

__global__ void seqlocks_kernel_strong(unsigned int * seqlock,
                                       int * dataArr0,
                                       int * dataArr1,
                                       int * outArr,
                                       const int groupSize_seqlock) {
  // local variables
  // unique thread ID for each thread
  const unsigned int threadID = ((blockIdx.x * blockDim.x) + threadIdx.x);
  // need 0th thread in each half wave to be master
  const bool isMasterThread = (threadIdx.x == 0);
  const unsigned int dataLoc = (threadIdx.x % WAVE_SIZE);
  // determine which seqlock I am accessing
  const unsigned int seqlockLoc = (blockIdx.x % groupSize_seqlock);

  // iterate a few times to provide more accesses, reuse, etc.
  for (int i = 0; i < NUM_ITERS; ++i) {
    // 1/16 WGs is a writer, rest are readers
    if (blockIdx.x % 16 == 0) {
      writer_strong(seqlock, dataArr0, dataArr1, seqlockLoc, dataLoc, isMasterThread);
    } else {
      reader_strong(seqlock, dataArr0, dataArr1, outArr, threadID, seqlockLoc, dataLoc, isMasterThread);
    }
  }
}

#endif
