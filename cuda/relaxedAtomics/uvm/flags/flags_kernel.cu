/*
  max exponential backoff value (need to make this desired
  power of 2 * 2 because we use bitwise ANDs of MAX_BACKOFF_EXP-1 to
  do the wraparound.
*/
#define MAX_BACKOFF_EXP 256

/*
  Main TB accesses some of its data, then sets the stop flag and joins a global
  barrier.  Once all TBs join the barrier, it checks the dirty flag and then
  accesses the dirty data in dataArr if the dirty flag is set.
*/
__device__ void mainTB(unsigned int * stop,
                       unsigned int * dirty,
                       unsigned int * outArr,
                       unsigned int * barrierBuffers,
                       unsigned int * perSMBarrierBuffers,
                       const int numAccsPerThr,
                       const int numRepeats,
                       const bool isMasterThread,
                       const unsigned int myBaseLoc,
                       const unsigned int numBlocksAtBarr,
                       const int smID,
                       const int perSM_blockID,
                       const int numTBs_perSM,
                       const int arrayStride,
                       const int maxBlocks) {
  // local variables
  volatile __shared__ unsigned int mainTBSum[256]; // local per TB counter
  __shared__ bool isDirty;

  if (isMasterThread) {
    isDirty = false;
  }
  __syncthreads();
  mainTBSum[threadIdx.x] = 0;
  __syncthreads();

  /*
    Step 1 -- read my data in the array a couple times so set isn't set
    immediately (each thread in the TB reads its own chunk, increments a
    scratchpad counter).
  */
  for (int i = 0; i < numRepeats; ++i) {
    for (int j = 0; j < numAccsPerThr; ++j) {
      // to make each accessed coalesced, need to access a location blockDim
      // away -- thus each thread accesses 1 location from each TBs data
      mainTBSum[threadIdx.x] += numAccsPerThr;
    }
    __syncthreads();
  }

  // Step 2 -- set stop to true (1) with one of the threads in the TB
  if (isMasterThread) { atomicAdd(stop, 1); }
  __syncthreads();

  /*
    Step 3 -- join barrier, use a tree barrier to reduce overhead
    all TBs on this SM do a local barrier first (reader is just one of the
    TBs on this SM for this barrier).
  */
  joinLFBarrier_helper(barrierBuffers, perSMBarrierBuffers, numBlocksAtBarr,
                       smID, perSM_blockID, numTBs_perSM, arrayStride,
                       maxBlocks);

  /*
    Step 4 -- once all TBs have joined the barrier, check dirty and access
    dirty data if it's set -- use an atomicAdd of 0 to check dirty because
    CUDA doesn't have an atomic load.  This load can be relaxed. Only have
    one thread in TB to check it to reduce contention.
  */
  if (isMasterThread) {
    isDirty = (atomicAdd(dirty, 0) != 0);
  }
  __syncthreads();

  if (isDirty) {
    // write to output array
    outArr[myBaseLoc] = mainTBSum[threadIdx.x];
  }
}

/*
  Worker TBs spin, waiting for main TB to set the stop flag.  While they are
  spinning, they potentially write their data in dataArr and set the dirty
  flag.
*/
__device__ void workerTBs(unsigned int * stop,
                          unsigned int * dirty,
                          unsigned int * barrierBuffers,
                          unsigned int * perSMBarrierBuffers,
                          float * randArr,
                          const int numAccsPerThr,
                          const bool isMasterThread,
                          const unsigned int myBaseLoc,
                          const unsigned int numBlocksAtBarr,
                          const int smID,
                          const int perSM_blockID,
                          const int numTBs_perSM,
                          const int arrayStride,
                          const float cutoffVal,
                          const unsigned int numIters,
                          const int maxBlocks) {
  // local variables
  unsigned int randTBLoc = (blockIdx.x * blockDim.x);
  int myNumIters = 0; // number of times I have executed the loop
  __shared__ unsigned int stopSet;
  __shared__ int backoff;
  volatile __shared__ unsigned int dataArr[256];
  const unsigned int numDummyComps = numAccsPerThr/4;

  /*
    Execute until stop is set (use atomicAdd + 0 because CUDA doesn't
    have an atomic load) -- only want 1 thread per TB to check stop.
  */
  if (isMasterThread) {
    // start exponential backoff at 2 so if we only miss once we don't wait a
    // long time
    backoff = 2;
    stopSet = (atomicAdd(stop, 0) != 0);
  }
  __syncthreads();
  dataArr[threadIdx.x] = threadIdx.x;
  __syncthreads();

  while (!stopSet || (myNumIters < numIters)) {
    // Condition for setting dirty to true -- some TBs can write, depending on
    // their random value vs. the cutoff value
    if (randArr[randTBLoc] <= cutoffVal) {
      // do local computations so x and dirty not immediately set
      for (int i = 0; i < numDummyComps; ++i) { dataArr[threadIdx.x]++; }
      __syncthreads();

      /*
        Only 1 thread per TB needs to set dirty (set after we're done
        writing the data since dirty potentially orders requests).  Use
        an atomicAdd to increment dirty without a store release.  Thus,
        if dirty is > 0, it is "set".
      */
      if (isMasterThread) { atomicAdd(dirty, 1); }
      __syncthreads();
    } else {
      // if we've been waiting for a long time, wrap around and check stop
      // more frequently
      if (isMasterThread) {
        backoff = (backoff * 2) & (MAX_BACKOFF_EXP-1);
      }
      __syncthreads();

#if ((HAS_NANOSLEEP == 1) && (CUDART_VERSION >= 1100))
      __nanosleep(backoff);
#else
      // do local computations so stop isn't immediately checked
      for (int i = 0; i < backoff; ++i) { ; }
#endif
    }

    // check if stop is set for next loop iteration
    if (isMasterThread) {
      stopSet = (atomicAdd(stop, 0) != 0);
    }
    __syncthreads();

    ++myNumIters;
    randTBLoc += gridDim.x;
  }

  /*
    Once stop is set to true, we need to join the barrier (even though we're
    about to exit, still need to join the barrier so the main TB knows we're
    not still executing); use a tree barrier to reduce overhead all TBs on this
    SM do a local barrier first (reader may be one of the TBs on this SM for
    the local barrier, but that's ok).
  */
  joinLFBarrier_helper(barrierBuffers, perSMBarrierBuffers, numBlocksAtBarr,
                       smID, perSM_blockID, numTBs_perSM, arrayStride, maxBlocks);
}

__global__ void flags_kernel(unsigned int * barrierBuffers,
                             unsigned int * perSMBarrierBuffers,
                             unsigned int * stop,
                             unsigned int * dirty,
                             unsigned int * outArr,
                             float * randArr,
                             const int arrayStride,
                             const int numAccsPerThr,
                             const int numRepeats,
                             const float cutoffVal,
                             const unsigned int numIters,
                             const int maxBlocks,
                             const int numSMs) {
  // local variables
  const bool isMainTB = (blockIdx.x == (gridDim.x - 1)); // last TB is main TB
  const bool isMasterThread = (threadIdx.x == 0);
  const unsigned int myBaseLoc = (blockIdx.x * blockDim.x) + threadIdx.x;
  // represents the number of TBs going to the barrier (max numSMs, gridDim.x if
  // fewer TBs than SMs).  Also represents the number of TBs there are.
  const unsigned int numBlocksAtBarr = ((gridDim.x < numSMs) ? gridDim.x : numSMs);
  const int smID = (blockIdx.x % numBlocksAtBarr); // mod by # SMs to get SM ID
  // determine if I'm TB 0 on my SM
  const int perSM_blockID = (blockIdx.x / numBlocksAtBarr);
  // given the gridDim.x, we can figure out how many TBs are on our SM -- assume
  // all SMs have an identical number of TBs
  /*
  ** NOTE: At the moment the lock-free barrier only works when there are:
    a) the same number of TBs per SM or
    b) fewer TBs than there are SMs.

    This is because the numTBs_perSM does not vary per SM, so the SMs that have
    more TBs than other SMs can't perform their local barriers correctly.  Need
    to fix this to make it more portable.
  */
  int numTBs_perSM = (gridDim.x / numBlocksAtBarr);
  if (numTBs_perSM == 0) { ++numTBs_perSM; } // always have to have at least 1

  /*
    The barrier is not used to separate any global data accesses (the only
    global access is a store after the barrier from the main TB), so list
    all regions but really nothing to invalidate.
  */

  /*
    main TB accesses some of its data, then sets the stop flag and joins a
    global barrier.  Once all TBs join the barrier, it checks the dirty flag
    and then accesses x if the dirty flag is set.
  */
  if (isMainTB) {
    mainTB(stop, dirty, outArr, barrierBuffers, perSMBarrierBuffers,
           numAccsPerThr, numRepeats, isMasterThread, myBaseLoc, numBlocksAtBarr,
           smID, perSM_blockID, numTBs_perSM, arrayStride, maxBlocks);
  }
  /*
    worker TBs spin, waiting for main TB to set the stop flag.  While they are
    spinning, they potentially write their data in local dataArr and set the
    dirty and x flags.
  */
  else {
    workerTBs(stop, dirty, barrierBuffers, perSMBarrierBuffers, randArr,
              numAccsPerThr, isMasterThread, myBaseLoc, numBlocksAtBarr, smID,
              perSM_blockID, numTBs_perSM, arrayStride, cutoffVal, numIters, maxBlocks);
  }
}
