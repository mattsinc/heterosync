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
inline __device__ void reader_tfs(unsigned int * seqlock,
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
    // atomic load, need acquire semantics so use threadfence since in relaxed
    // atomics region -- use TF for this
    __threadfence();
    if (isMasterThread) {
      seq0 = atomicAdd(&(seqlock[seqlockLoc]), 0);
    }
    __syncthreads();

    r1 = atomicExch(&(dataArr0[dataLoc]), 0);
    r2 = atomicExch(&(dataArr1[dataLoc]), 0);

    // use TF to simulate release semantics
    __threadfence();
    if (isMasterThread) {
      seq1 = atomicAdd(&(seqlock[seqlockLoc]), 0);
    }
    __syncthreads();
  } while ((seq0 != seq1) || (seq0 & 1));

  // update my threads location in the out array with data array values
  // just use the combination of the data array values
  outArr[threadID] = r1 + r2;
}

inline __device__ void reader(unsigned int * seqlock,
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
    if (isMasterThread) {
      seq0 = atomicAdd(&(seqlock[seqlockLoc]), 0);
    }
    __syncthreads();

    r1 = atomicExch(&(dataArr0[dataLoc]), 0);
    r2 = atomicExch(&(dataArr1[dataLoc]), 0);

    if (isMasterThread) {
      seq1 = atomicAdd(&(seqlock[seqlockLoc]), 0);
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
inline __device__ void writer_tfs(unsigned int * seqlock,
                                  int * dataArr0,
                                  int * dataArr1,
                                  const unsigned int seqlockLoc,
                                  const unsigned int dataLoc,
                                  const bool isMasterThread) {
  // local variables
  volatile int backoff = 1;
  unsigned int seq0 = 0;

  if (isMasterThread) {
    // use TFs to simulate acquire semantics, since seqlock is in relaxed
    // atomic region
    __threadfence();
    seq0 = atomicAdd(&(seqlock[seqlockLoc]), 0);

    // spin until no readers or other writers
    while ((seq0 & 1) ||
           (atomicCAS(&(seqlock[seqlockLoc]), seq0, seq0+1) != seq0)) {
      // if we failed, wait for a little while before trying again
      sleepFunc(backoff);
      // backoff so don't keep hammering seqlock
      backoff = ((backoff << 1) + 1) & (MAX_BACKOFF-1);

      // reload seq0 to see any changes the other writers have made
      // use TFs to simulate acquire semantics, since seqlock is in relaxed
      // atomic region
      __threadfence();
      seq0 = atomicAdd(&(seqlock[seqlockLoc]), 0);
    }
  }
  __syncthreads();
  // need threadfence here to handle the compare_exchange_weak semantics -- need
  // a fence if it succeeds
  __threadfence();

  /*
    Write the current sequence number, value doesn't really matter.
    These accesses can be overlapped.
  */
  atomicExch(&(dataArr0[dataLoc]), seq0);
  atomicExch(&(dataArr1[dataLoc]), seq0);

  // use TFs to simulate release semantics, since seqlock is in relaxed atomic
  // region
  __threadfence();
  if (isMasterThread) {
    atomicExch(&(seqlock[seqlockLoc]), seq0 + 2);
  }
  __syncthreads();
}

inline __device__ void writer(unsigned int * seqlock,
                              int * dataArr0,
                              int * dataArr1,
                              const unsigned int seqlockLoc,
                              const unsigned int dataLoc,
                              const bool isMasterThread) {
  // local variables
  volatile int backoff = 1;
  unsigned int seq0 = 0;

  if (isMasterThread) {
    seq0 = atomicAdd(&(seqlock[seqlockLoc]), 0);

    while ((seq0 & 1) || (atomicCAS(&(seqlock[seqlockLoc]), seq0, seq0+1) != seq0)) {
      // if we failed, wait for a little while before trying again
      sleepFunc(backoff);
      // backoff so don't keep hammering seqlock
      backoff = ((backoff << 1) + 1) & (MAX_BACKOFF-1);

      // reload seq0 to see any changes the other writers have made
      seq0 = atomicAdd(&(seqlock[seqlockLoc]), 0);
    }
  }
  __syncthreads();

  /*
    Write the current sequence number, value doesn't really matter.
    These accesses can be overlapped.  
  */
  atomicExch(&(dataArr0[dataLoc]), seq0);
  atomicExch(&(dataArr1[dataLoc]), seq0);

  if (isMasterThread) {
    atomicExch(&(seqlock[seqlockLoc]), seq0 + 2);
  }
  __syncthreads();
}

__global__ void seqlocks_kernel(unsigned int * seqlock,
                                int * dataArr0,
                                int * dataArr1,
                                int * outArr,
                                const int groupSize_seqlock/*,
                                const region_t seqReg,
                                const region_t dataReg,
                                const region_t outReg*/) {
  // local variables
  // unique thread ID for each thread
  const unsigned int threadID = ((blockIdx.x * blockDim.x) + threadIdx.x);
  const bool isMasterThread = (threadIdx.x == 0);
  const unsigned int dataLoc = (threadIdx.x % WAVE_SIZE);
  // determine which seqlock I am accessing
  const unsigned int seqlockLoc = (blockIdx.x % groupSize_seqlock);

  // iterate a few times to provide more accesses, reuse, etc.
  for (int i = 0; i < NUM_ITERS; ++i) {
    // 1/16 WGs is a writer, rest are readers
    if (blockIdx.x % 16 == 0) {
      writer(seqlock, dataArr0, dataArr1, seqlockLoc, dataLoc, isMasterThread);
    } else {
      reader(seqlock, dataArr0, dataArr1, outArr, threadID, seqlockLoc, dataLoc, isMasterThread);
    }
  }
}

__global__ void seqlocks_kernel_tfs(unsigned int * seqlock,
                                    int * dataArr0,
                                    int * dataArr1,
                                    int * outArr,
                                    const int groupSize_seqlock/*,
                                    const region_t seqReg,
                                    const region_t dataReg,
                                    const region_t outReg*/) {
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
      writer_tfs(seqlock, dataArr0, dataArr1, seqlockLoc, dataLoc, isMasterThread);
    } else {
      reader_tfs(seqlock, dataArr0, dataArr1, outArr, threadID, seqlockLoc, dataLoc, isMasterThread);
    }
  }
}

#endif
