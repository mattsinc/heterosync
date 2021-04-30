#define WARP_SIZE 32
#define HALF_WARP_SIZE (WARP_SIZE >> 1)
#define NUM_ITERS 10

/*
  max exponential backoff value (need to make this desired
  power of 2 * 2 because we use bitwise ANDs of MAX_BACKOFF-1 to
  do the wraparound.
*/
#define MAX_BACKOFF 32

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

    /*
    r1 = atomicAdd(&(dataArr0[dataLoc]), 0);
    r2 = atomicAdd(&(dataArr1[dataLoc]), 0);
    */
    /*
      Replace the above atomics with inlined PTX to ensure that they are
      next to each other in the instruction sequence and thus can be
      overlapped.  Need separate blocks of assembly because can't have
      two output registers in a single assembly block.

      NOTE: Across all of the inlined assembly blocks we can't reuse the
      same temp reg names.
    */
    int * data1Addr = &(dataArr0[dataLoc]);
    int * data2Addr = &(dataArr1[dataLoc]);
    asm volatile(// Temp Registers
                 // PTX Instructions
                 "atom.or.b32 %0, [%1], 0;\n\t"
                 // outputs
                 : "=r"(r1)
                 // inputs
                 : "l"(data1Addr)
                 );
    asm volatile(// Temp Registers
                 // PTX Instructions
                 "atom.or.b32 %0, [%1], 0;\n\t"
                 // outputs
                 : "=r"(r2)
                 // inputs
                 : "l"(data2Addr)
                 );

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

    /*
    r1 = atomicAdd(&(dataArr[0]), 0);
    r2 = atomicAdd(&(dataArr[1]), 0);
    */
    /*
      Replace the above atomics with inlined PTX to ensure that they are
      next to each other in the instruction sequence and thus can be
      overlapped.  Need separate blocks of assembly because can't have
      two output registers in a single assembly block.

      NOTE: Across all of the inlined assembly blocks we can't reuse the
      same temp reg names.
    */
    int * data1Addr = &(dataArr0[dataLoc]);
    int * data2Addr = &(dataArr1[dataLoc]);
    asm volatile(// Temp Registers
                 // PTX Instructions
                 "atom.or.b32 %0, [%1], 0;\n\t"
                 // outputs
                 : "=r"(r1)
                 // inputs
                 : "l"(data1Addr)
                 );
    asm volatile(// Temp Registers
                 // PTX Instructions
                 "atom.or.b32 %0, [%1], 0;\n\t"
                 // outputs
                 : "=r"(r2)
                 // inputs
                 : "l"(data2Addr)
                 );

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
      // backoff so don't keep hammering seqlock
      backoff = ((backoff << 1) + 1) & (MAX_BACKOFF-1);
      for (int i = 0; i < backoff; ++i) { ; }

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
    atomicOr is a reprogrammed (unpaired) atomic store, but these accesses can
    be overlapped.  Write the current sequence number, value doesn't really
    matter.
  */
  /*
  atomicOr(&(dataArr0[dataLoc]), seq0);
  atomicOr(&(dataArr1[dataLoc]), seq0);
  */
  /*
    Replace the above atomics with inlined PTX to ensure that they are
    next to each other in the instruction sequence and thus can be
    overlapped (only the data accesses can be overlapped).

    NOTE: Across all of the inlined assembly blocks we can't reuse the
    same temp reg names.
  */
  int * data1Addr = &(dataArr0[dataLoc]);
  int * data2Addr = &(dataArr1[dataLoc]);
  asm volatile(// Temp Registers
               ".reg .s32 q0;\n\t"      // temp reg q0 (seq0)
               ".reg .s32 q1;\n\t"      // temp reg q1 (atomOr(dataAddr1) result)
               ".reg .s32 q2;\n\t"      // temp reg q2 (atomOr(dataAddr2) result)
               // PTX Instructions
               "mov.s32 q0, %0;\n\t"
               "atom.or.b32 q1, [%1], q0;\n\t"
               "atom.or.b32 q2, [%2], q0;"
               // no outputs
               // inputs
               :: "r"(seq0), "l"(data1Addr), "l"(data2Addr)
               );

  // use TFs to simulate release semantics, since seqlock is in relaxed atomic
  // region
  __threadfence();
  if (isMasterThread) {
    atomicOr(&(seqlock[seqlockLoc]), seq0 + 2);
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
      // backoff so don't keep hammering seqlock
      backoff = ((backoff << 1) + 1) & (MAX_BACKOFF-1);
      for (int i = 0; i < backoff; ++i) { ; }

      // reload seq0 to see any changes the other writers have made
      seq0 = atomicAdd(&(seqlock[seqlockLoc]), 0);
    }
  }
  __syncthreads();

  /*
    atomicOr is a reprogrammed (unpaired) atomic store, but these accesses can
    be overlapped.  Write the current sequence number, value doesn't really
    matter.
  */
  /*
  atomicOr(&(dataArr0[dataLoc]), seq0);
  atomicOr(&(dataArr1[dataLoc]), seq0);
  */
  /*
    Replace the above atomics with inlined PTX to ensure that they are
    next to each other in the instruction sequence and thus can be
    overlapped (only the data accesses can be overlapped).

    NOTE: Across all of the inlined assembly blocks we can't reuse the
    same temp reg names.
  */
  int * data1Addr = &(dataArr0[dataLoc]);
  int * data2Addr = &(dataArr1[dataLoc]);
  asm volatile(// Temp Registers
               ".reg .s32 q0;\n\t"      // temp reg q0 (seq0)
               ".reg .s32 q1;\n\t"      // temp reg q1 (atomOr(dataAddr1) result)
               ".reg .s32 q2;\n\t"      // temp reg q2 (atomOr(dataAddr2) result)
               // PTX Instructions
               "mov.s32 q0, %0;\n\t"
               "atom.or.b32 q1, [%1], q0;\n\t"
               "atom.or.b32 q2, [%2], q0;"
               // no outputs
               // inputs
               :: "r"(seq0), "l"(data1Addr), "l"(data2Addr)
               );

  if (isMasterThread) {
    atomicOr(&(seqlock[seqlockLoc]), seq0 + 2);
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
  const unsigned int dataLoc = (threadIdx.x % WARP_SIZE);
  // determine which seqlock I am accessing
  const unsigned int seqlockLoc = (blockIdx.x % groupSize_seqlock);

  /*
  if (threadIdx.x == 0) {
    __denovo_setAcquireRegion(dataReg); // should only be written with atomics
    __denovo_setAcquireRegion(outReg);
    __denovo_addAcquireRegion(seqReg); // should only be written with atomics
  }
  __syncthreads();
  */

  // iterate a few times to provide more accesses, reuse, etc.
  for (int i = 0; i < NUM_ITERS; ++i) {
    // 1/16 TBs is a writer, rest are readers
    if (blockIdx.x % 16 == 0) {
      writer(seqlock, dataArr0, dataArr1, seqlockLoc, dataLoc, isMasterThread);
    } else {
      reader(seqlock, dataArr0, dataArr1, outArr, threadID, seqlockLoc, dataLoc, isMasterThread);
    }
  }

  /*
  if (threadIdx.x == 0) {
    __denovo_gpuEpilogue(seqReg);   // written with atomics (reader, writer)
    __denovo_gpuEpilogue(dataReg);  // written in reader
    __denovo_gpuEpilogue(outReg);   // written with atomics (reader, writer)
  }
  */
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
  // need 0th thread in each half warp to be master
  const bool isMasterThread = (threadIdx.x == 0);
  const unsigned int dataLoc = (threadIdx.x % WARP_SIZE);
  // determine which seqlock I am accessing
  const unsigned int seqlockLoc = (blockIdx.x % groupSize_seqlock);

  /*
  if (threadIdx.x == 0) {
    __denovo_setAcquireRegion(dataReg); // should only be written with atomics
    __denovo_setAcquireRegion(outReg);
    __denovo_addAcquireRegion(seqReg); // should only be written with atomics
  }
  __syncthreads();
  */

  // iterate a few times to provide more accesses, reuse, etc.
  for (int i = 0; i < NUM_ITERS; ++i) {
    // 1/16 TBs is a writer, rest are readers
    if (blockIdx.x % 16 == 0) {
      writer_tfs(seqlock, dataArr0, dataArr1, seqlockLoc, dataLoc, isMasterThread);
    } else {
      reader_tfs(seqlock, dataArr0, dataArr1, outArr, threadID, seqlockLoc, dataLoc, isMasterThread);
    }
  }

  /*
  if (threadIdx.x == 0) {
    __denovo_gpuEpilogue(seqReg);   // written with atomics (reader, writer)
    __denovo_gpuEpilogue(dataReg);  // written in reader
    __denovo_gpuEpilogue(outReg);   // written with atomics (reader, writer)
  }
  */
}
