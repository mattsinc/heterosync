#define WARP_SIZE 32
#define HALF_WARP_SIZE (WARP_SIZE >> 1)

__global__ void refCounter_kernel(unsigned int * d_counters0,
                                  unsigned int * d_counters1,
                                  unsigned int * d_del0,
                                  unsigned int * d_del1,
                                  const unsigned int numRepeats,
                                  const unsigned int numSharersPerGroup,
                                  const unsigned int numCounters,
                                  const unsigned int numSharingGroups,
                                  const unsigned int numCounters_perSharingGroup) {
  // local variables
  const unsigned int myBaseLoc = ((blockIdx.x * blockDim.x) + threadIdx.x);
  const unsigned int mySharingGroup = (blockIdx.x % numSharingGroups);
  const unsigned int myCounterLoc = ((mySharingGroup * numCounters_perSharingGroup) + threadIdx.x);
  unsigned int * counterAddr0, * counterAddr1;
  __shared__ volatile int dummyLocal[256]; // for doing local dummy calculations, assumes blockDim.x <= 256

  dummyLocal[threadIdx.x] = 0;
  __syncthreads();

  // the counters each thread accesses is fixed, regardless of the number of loop iterations
  counterAddr0 = &(d_counters0[myCounterLoc]);
  counterAddr1 = &(d_counters1[myCounterLoc]);

  // repeat this process a few times
  for (int i = 0; i < numRepeats; ++i) {
    // these atomics can be reordered with each other
    atomicAdd(counterAddr0, 1);
    atomicAdd(counterAddr1, 1);

    // Do accesses in scratchpad here to space inc and dec out
    for (int j = 0; j < numRepeats * 2; ++j) {
      dummyLocal[threadIdx.x] += j;
      __syncthreads();
    }

    // If the shared counter == 0 (old value == 1), then mark the "object" to
    // be deleted
    // use atomicDec's with threadfences to ensure that we have acquire-release
    // semantics for DRF1 and DRF0
    unsigned int currCount0 = atomicDec(counterAddr0, 1000000000);
    __threadfence();
    unsigned int currCount1 = atomicDec(counterAddr1, 1000000000);
    __threadfence();
    if (currCount0 <= 1) {
      d_del0[myBaseLoc] = true;
    }

    if (currCount1 <= 1) {
      d_del1[myBaseLoc] = true;
    }
  }
}
