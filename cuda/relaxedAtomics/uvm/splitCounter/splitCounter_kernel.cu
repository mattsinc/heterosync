#define WARP_SIZE 32
#define HALF_WARP_SIZE (WARP_SIZE >> 1)
#define INC_VAL 1

/*
  add_split_counter updates my TBs counter with the inputted value.  Since
  multiple threads are accessing the same counter, we need to use atomics to
  update the counter.
 */
inline __device__ void add_split_counter(unsigned int * counters,
                                         const unsigned int newVal,
                                         const unsigned int numCounters_perTB,
                                         const unsigned int warpID) {
  // local variables
  // each thread in each TB starts at a different location in the counter array
  // to ensure that we have ordering paths (which means quantum races may occur)
  const unsigned int loc = ((blockIdx.x * numCounters_perTB) + // number of counters in TBs with smaller block IDs
                            (warpID * HALF_WARP_SIZE) + // number of counters in warps from this TB with smaller warp IDs
                            (threadIdx.x % HALF_WARP_SIZE)); // within my warp, which counter should I start with

  // this atomic accesses can be relaxed
  atomicAdd(&(counters[loc]), newVal);
}

/*
  read_split_counter returns an approximation of the current split count
  value.  If we wanted to get a precise value, we would need to use atomic
  accesses.  However, split counter is amenable to approximate values, so we
  instead access an array of random values and return that.
*/
inline __device__ unsigned int read_split_counter(unsigned int * counters,
                                                  const unsigned int numCounters_perTB,
                                                  const unsigned int warpID) {
  // local variables
  __shared__ unsigned int localSum[256]; // assumes blockDim.x <= 256
  const unsigned int numCounters = gridDim.x * numCounters_perTB;
  // this assumes that the numCounters is evenly divisible by the TB size
  // only half the threads in the TB do the reads, so divide by 2 to make sure
  // that all elements are accessed
  const unsigned int numAccsPerThr = (numCounters / (blockDim.x >> 1));
  // each thread in each TB starts at a different location in the counter array
  // to ensure that we have ordering paths (which means quantum races may occur)
  const unsigned int myBaseLoc = ((numCounters_perTB * blockIdx.x) + // starting points for my TB
                                  ((numAccsPerThr * HALF_WARP_SIZE) * warpID) + // starting points for warps in my TB with smaller warp IDs
                                  (threadIdx.x % HALF_WARP_SIZE)); // starting point within my warp
  // start each chain of accesses at my location in the array
  unsigned int currLoc = (myBaseLoc % numCounters);

  localSum[threadIdx.x] = 0; // each thread initializes its local sum to 0
  __syncthreads();

  /*
    Each thread takes a chunk of the counters and adds them into a scratchpad
    array -- since CUDA doesn't have an atomic load, do an add of 0 to get the
    current counter value (these can be relaxed).
  */
  for (int i = 0; i < numAccsPerThr; ++i) {
    localSum[threadIdx.x] += atomicAdd(&(counters[currLoc]), 0);
    __syncthreads();

    // the next access is numCounters_perTB away to coalesce the accesses in a
    // half-warp in a given iteration
    currLoc = ((currLoc + numCounters_perTB) % numCounters);
  }

  // to get the final sum we need to sum the local partial sums -- do reduction
  // in scratchpad too (from CUDA SDK reduction example)
  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    if ((threadIdx.x % (2*s)) == 0) {
      localSum[threadIdx.x] += localSum[threadIdx.x + s];
    }
    __syncthreads();
  }

  return localSum[0];
}

__global__ void splitCounter_kernel(unsigned int * counters,
                                    unsigned int * outArr) {
  // local variables
  // every other thread in each TB has its own counter (so readers and writers
  // access the same counters) -- assumes all TBs are equally sized and powers
  // of 2
  const unsigned int numCounters_perTB = (blockDim.x >> 1);
  // get my warp ID within my TB
  const unsigned int warpID = (threadIdx.x / WARP_SIZE);
  // unique thread ID for each thread
  const unsigned int threadID = ((blockIdx.x * blockDim.x) + threadIdx.x);

  // odd half-warps in each TB update the split counter, even half-warps in each
  // TB read the split counter
  if ((threadIdx.x % WARP_SIZE) < HALF_WARP_SIZE) {
    add_split_counter(counters, INC_VAL, numCounters_perTB, warpID);
  } else {
    //outArr[threadID] += read_split_counter(counters, numCounters_perTB, warpID);
    /*
      Write to outArr instead of reading and writing because the reads affect
      the stall profile.  If we run the kernel multiple times, then the outArr
      value will just be the value read in the last iteration of the kernel.
    */
    outArr[threadID] = read_split_counter(counters, numCounters_perTB, warpID);
  }
}
