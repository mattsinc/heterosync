#include <cstdio>
#include <string>
#include <assert.h>
#include <math.h>
#include "cuda_error.h"

#define NUM_THREADS_PER_BLOCK 32
#define MAD_MUL 1.1f
#define MAD_ADD 0.25f
#define NUM_WORDS_PER_CACHELINE 16
#define NUM_THREADS_PER_HALFWARP 16

// separate .cu files
#include "cudaLocks.cu"
#include "cudaLocksBarrier.cu"
#include "cudaLocksMutex.cu"
#include "cudaLocksSemaphore.cu"

// program globals
const int NUM_REPEATS = 10;
int NUM_LDST = 0;
int numTBs = 0;
// number of SMs our GPU has
int NUM_SM = 0;
int MAX_BLOCKS = 0;

bool pageAlign = false;

/*
  Helper function to do data accesses for golden checking code.
*/
void accessData_golden(float * storageGolden, int currLoc, int numStorageLocs)
{
  /*
    If this location isn't the first location accessed by a
    thread, update it -- each half-warp accesses (NUM_LDST + 1) cache
    lines.
  */
  if (currLoc % (NUM_THREADS_PER_HALFWARP * (NUM_LDST + 1)) >=
      NUM_WORDS_PER_CACHELINE)
  {
    assert((currLoc - NUM_WORDS_PER_CACHELINE) >= 0);
    assert(currLoc < numStorageLocs);
    // each location is dependent on the location accessed at the
    // same word on the previous cache line
    storageGolden[currLoc] =
      ((storageGolden[currLoc -
                      NUM_WORDS_PER_CACHELINE]/* * MAD_MUL*/) /*+ MAD_ADD*/);
  }
}

/*
  Shared function that does the critical section data accesses for the barrier
  and mutex kernels.

  NOTE: kernels that access data differently (e.g., semaphores) should not call
  this function.
*/
inline __device__ void accessData(float * storage, int threadBaseLoc,
                                  int threadOffset, int NUM_LDST)
{
  // local variables
  int readLoc = 0, writeLoc = 0;

  for (int n = NUM_LDST-1; n >= 0; --n) {
    writeLoc = ((threadBaseLoc + n + 1) * NUM_WORDS_PER_CACHELINE) +
               threadOffset;
    readLoc = ((threadBaseLoc + n) * NUM_WORDS_PER_CACHELINE) + threadOffset;
    storage[writeLoc] = ((storage[readLoc]/* * MAD_MUL*/) /*+ MAD_ADD*/);
  }
}

/*
  Helper function for semaphore writers.  Although semaphore writers are
  iterating over all locations accessed on a given SM, the logic for this
  varies and is done outside this helper, so can just call accessData().
*/
inline __device__ void accessData_semWr(float * storage, int threadBaseLoc,
                                        int threadOffset, int NUM_LDST)
{
  accessData(storage, threadBaseLoc, threadOffset, NUM_LDST);
}


/*
  Helper function for semaphore readers.
*/
inline __device__ void accessData_semRd(float * storage,
                                        volatile float * dummyArray,
                                        int threadBaseLoc,
                                        int threadOffset, int NUM_LDST)
{
  for (int n = NUM_LDST-1; n >= 0; --n) {
    dummyArray[threadIdx.x] +=
      storage[((threadBaseLoc + n) * NUM_WORDS_PER_CACHELINE) +
              threadOffset];
    __syncthreads();
  }
}

// performs a tree barrier.  Each TB on an SM accesses unique data then joins a
// local barrier.  1 of the TBs from each SM then joins the global barrier
__global__ void kernelAtomicTreeBarrierUniq(float * storage,
                                            cudaLockData_t * gpuLockData,
                                            unsigned int * perSMBarrierBuffers,
                                            const int ITERATIONS,
                                            const int NUM_LDST,
                                            const int NUM_SM,
                                            const int MAX_BLOCKS)
{
  // local variables
  // thread 0 is master thread
  const bool isMasterThread = ((threadIdx.x == 0) && (threadIdx.y == 0) &&
                               (threadIdx.z == 0));
  // represents the number of TBs going to the barrier (max NUM_SM, gridDim.x if
  // fewer TBs than SMs).
  const unsigned int numBlocksAtBarr = ((gridDim.x < NUM_SM) ? gridDim.x :
                                        NUM_SM);
  const int smID = (blockIdx.x % numBlocksAtBarr); // mod by # SMs to get SM ID
  // all thread blocks on the same SM access unique locations because the
  // barrier can't ensure DRF between TBs
  int currBlockID = blockIdx.x;
  int tid = ((currBlockID * blockDim.x) + threadIdx.x);
  // want striding to happen across cache lines so that each thread in a
  // half-warp accesses sequential words
  int threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  int threadOffset = (threadIdx.x % NUM_WORDS_PER_CACHELINE);
  // determine if I'm TB 0 on my SM
  const int perSM_blockID = (blockIdx.x / numBlocksAtBarr);
  // given the gridDim.x, we can figure out how many TBs are on our SM -- assume
  // all SMs have an identical number of TBs
  int numTBs_perSM = (gridDim.x / numBlocksAtBarr);
  if (numTBs_perSM == 0) { ++numTBs_perSM; } // always have to have at least 1

  for (int i = 0; i < ITERATIONS; ++i)
  {
    accessData(storage, threadBaseLoc, threadOffset, NUM_LDST);

    joinBarrier_helper(gpuLockData->barrierBuffers, perSMBarrierBuffers,
                       numBlocksAtBarr, smID, perSM_blockID, numTBs_perSM,
                       isMasterThread, MAX_BLOCKS);

    // get new thread ID by trading amongst TBs -- + 1 block ID to shift to next
    // SMs data
    currBlockID = ((currBlockID + 1) % gridDim.x);
    tid = ((currBlockID * blockDim.x) + threadIdx.x);
    threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  }
}

// like the tree barrier but also has TBs exchange work locally before joining
// the global barrier
__global__ void kernelAtomicTreeBarrierUniqLocalExch(float * storage,
                                                     cudaLockData_t * gpuLockData,
                                                     unsigned int * perSMBarrierBuffers,
                                                     const int ITERATIONS,
                                                     const int NUM_LDST,
                                                     const int NUM_SM,
                                                     const int MAX_BLOCKS)
{
  // local variables
  // thread 0 is master thread
  const bool isMasterThread = ((threadIdx.x == 0) && (threadIdx.y == 0) &&
                               (threadIdx.z == 0));
  // represents the number of TBs going to the barrier (max NUM_SM, gridDim.x if
  // fewer TBs than SMs).
  const unsigned int numBlocksAtBarr = ((gridDim.x < NUM_SM) ? gridDim.x :
                                        NUM_SM);
  const int smID = (blockIdx.x % numBlocksAtBarr); // mod by # SMs to get SM ID
  // all thread blocks on the same SM access unique locations because the
  // barrier can't ensure DRF between TBs
  int currBlockID = blockIdx.x;
  int tid = ((currBlockID * blockDim.x) + threadIdx.x);
  // want striding to happen across cache lines so that each thread in a
  // half-warp accesses sequential words
  int threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  int threadOffset = (threadIdx.x % NUM_WORDS_PER_CACHELINE);
  // determine if I'm TB 0 on my SM
  const int perSM_blockID = (blockIdx.x / numBlocksAtBarr);
  // given the gridDim.x, we can figure out how many TBs are on our SM -- assume
  // all SMs have an identical number of TBs
  int numTBs_perSM = (gridDim.x / numBlocksAtBarr);
  if (numTBs_perSM == 0) { ++numTBs_perSM; } // always have to have at least 1

  for (int i = 0; i < ITERATIONS; ++i)
  {
    accessData(storage, threadBaseLoc, threadOffset, NUM_LDST);

    // all TBs on this SM do a local barrier (if > 1 TB)
    if (numTBs_perSM > 1) {
      cudaBarrierAtomicLocal(perSMBarrierBuffers, smID, numTBs_perSM,
                             isMasterThread, MAX_BLOCKS);

      // exchange data within the TBs on this SM, then do some more computations
      currBlockID = ((currBlockID + numBlocksAtBarr) % gridDim.x);
      tid = ((currBlockID * blockDim.x) + threadIdx.x);
      threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));

      accessData(storage, threadBaseLoc, threadOffset, NUM_LDST);
    }

    joinBarrier_helper(gpuLockData->barrierBuffers, perSMBarrierBuffers,
                       numBlocksAtBarr, smID, perSM_blockID, numTBs_perSM,
                       isMasterThread, MAX_BLOCKS);

    // get new thread ID by trading amongst TBs -- + 1 block ID to shift to next
    // SMs data
    currBlockID = ((currBlockID + 1) % gridDim.x);
    tid = ((currBlockID * blockDim.x) + threadIdx.x);
    threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  }
}

// performs a tree barrier like above but with a lock-free barrier
__global__ void kernelFBSTreeBarrierUniq(float * storage,
                                         cudaLockData_t * gpuLockData,
                                         unsigned int * perSMBarrierBuffers,
                                         const int ITERATIONS,
                                         const int NUM_LDST,
                                         const int NUM_SM,
                                         const int MAX_BLOCKS)
{
  // local variables
  // represents the number of TBs going to the barrier (max NUM_SM, gridDim.x if
  // fewer TBs than SMs).
  const unsigned int numBlocksAtBarr = ((gridDim.x < NUM_SM) ? gridDim.x :
                                        NUM_SM);
  const int smID = (blockIdx.x % numBlocksAtBarr); // mod by # SMs to get SM ID
  // all thread blocks on the same SM access unique locations because the
  // barrier can't ensure DRF between TBs
  int currBlockID = blockIdx.x;
  int tid = ((currBlockID * blockDim.x) + threadIdx.x);
  // want striding to happen across cache lines so that each thread in a
  // half-warp accesses sequential words
  int threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  const int threadOffset = (threadIdx.x % NUM_WORDS_PER_CACHELINE);
  // determine if I'm TB 0 on my SM
  const int perSM_blockID = (blockIdx.x / numBlocksAtBarr);
  // given the gridDim.x, we can figure out how many TBs are on our SM -- assume
  // all SMs have an identical number of TBs
  int numTBs_perSM = (gridDim.x/numBlocksAtBarr);
  if (numTBs_perSM == 0) { ++numTBs_perSM; } // always have to have at least 1

  for (int i = 0; i < ITERATIONS; ++i)
  {
    accessData(storage, threadBaseLoc, threadOffset, NUM_LDST);

    joinLFBarrier_helper(gpuLockData->barrierBuffers, perSMBarrierBuffers,
                         numBlocksAtBarr, smID, perSM_blockID, numTBs_perSM,
                         gpuLockData->arrayStride, MAX_BLOCKS);

    // get new thread ID by trading amongst TBs -- + 1 block ID to shift to next
    // SMs data
    currBlockID = ((currBlockID + 1) % gridDim.x);
    tid = ((currBlockID * blockDim.x) + threadIdx.x);
    threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  }
}

// performs a tree barrier like above but with a lock-free barrier and has TBs
// exchange work locally before joining the global barrier
__global__ void kernelFBSTreeBarrierUniqLocalExch(float * storage,
                                                  cudaLockData_t * gpuLockData,
                                                  unsigned int * perSMBarrierBuffers,
                                                  const int ITERATIONS,
                                                  const int NUM_LDST,
                                                  const int NUM_SM,
                                                  const int MAX_BLOCKS)
{
  // local variables
  // represents the number of TBs going to the barrier (max NUM_SM, gridDim.x if
  // fewer TBs than SMs).
  const unsigned int numBlocksAtBarr = ((gridDim.x < NUM_SM) ? gridDim.x : NUM_SM);
  const int smID = (blockIdx.x % numBlocksAtBarr); // mod by # SMs to get SM ID
  // all thread blocks on the same SM access unique locations because the
  // barrier can't ensure DRF between TBs
  int currBlockID = blockIdx.x;
  int tid = ((currBlockID * blockDim.x) + threadIdx.x);
  // want striding to happen across cache lines so that each thread in a
  // half-warp accesses sequential words
  int threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  const int threadOffset = (threadIdx.x % NUM_WORDS_PER_CACHELINE);
  // determine if I'm TB 0 on my SM
  const int perSM_blockID = (blockIdx.x / numBlocksAtBarr);
  // given the gridDim.x, we can figure out how many TBs are on our SM -- assume
  // all SMs have an identical number of TBs
  int numTBs_perSM = (gridDim.x/numBlocksAtBarr);
  if (numTBs_perSM == 0) { ++numTBs_perSM; } // always have to have at least 1

  for (int i = 0; i < ITERATIONS; ++i)
  {
    accessData(storage, threadBaseLoc, threadOffset, NUM_LDST);

    // all TBs on this SM do a local barrier (if > 1 TB per SM)
    if (numTBs_perSM > 1) {
      cudaBarrierLocal(gpuLockData->barrierBuffers, numBlocksAtBarr,
                       gpuLockData->arrayStride, perSMBarrierBuffers, smID, numTBs_perSM,
                       perSM_blockID, false, MAX_BLOCKS);

      // exchange data within the TBs on this SM and do some more computations
      currBlockID = ((currBlockID + numBlocksAtBarr) % gridDim.x);
      tid = ((currBlockID * blockDim.x) + threadIdx.x);
      threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));

      accessData(storage, threadBaseLoc, threadOffset, NUM_LDST);
    }

    joinLFBarrier_helper(gpuLockData->barrierBuffers, perSMBarrierBuffers,
                         numBlocksAtBarr, smID, perSM_blockID, numTBs_perSM,
                         gpuLockData->arrayStride, MAX_BLOCKS);

    // get new thread ID by trading amongst TBs -- + 1 block ID to shift to next
    // SMs data
    currBlockID = ((currBlockID + 1) % gridDim.x);
    tid = ((currBlockID * blockDim.x) + threadIdx.x);
    threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  }
}

__global__ void kernelSleepingMutex(cudaMutex_t mutex, float * storage,
                                    cudaLockData_t * gpuLockData,
                                    const int ITERATIONS, const int NUM_LDST,
                                    const int NUM_SM)
{
  // local variables
  // all thread blocks access the same locations (rely on release to get
  // ownership in time)
  const int tid = threadIdx.x;
  // want striding to happen across cache lines so that each thread in a
  // half-warp accesses sequential words
  const int threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  const int threadOffset = (threadIdx.x % NUM_WORDS_PER_CACHELINE);
  __shared__ int myRingBufferLoc; // tracks my TBs location in the ring buffer

  if (threadIdx.x == 0) {
    myRingBufferLoc = -1; // initially I have no location
  }
  __syncthreads();

  for (int i = 0; i < ITERATIONS; ++i)
  {
    myRingBufferLoc = cudaMutexSleepLock(mutex, gpuLockData->mutexBuffers,
                                         gpuLockData->mutexBufferTails, gpuLockData->maxBufferSize,
                                         gpuLockData->arrayStride, NUM_SM);

    accessData(storage, threadBaseLoc, threadOffset, NUM_LDST);

    cudaMutexSleepUnlock(mutex, gpuLockData->mutexBuffers, myRingBufferLoc,
                         gpuLockData->maxBufferSize, gpuLockData->arrayStride, NUM_SM);
  }
}

__global__ void kernelSleepingMutexUniq(cudaMutex_t mutex, float * storage,
                                        cudaLockData_t * gpuLockData,
                                        const int ITERATIONS,
                                        const int NUM_LDST, const int NUM_SM)
{
  // local variables
  const int smID = (blockIdx.x % NUM_SM); // mod by # SMs to get SM ID
  // all thread blocks on the same SM access the same locations
  const int tid = ((smID * blockDim.x) + threadIdx.x);
  // want striding to happen across cache lines so that each thread in a
  // half-warp accesses sequential words
  const int threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  const int threadOffset = (threadIdx.x % NUM_WORDS_PER_CACHELINE);
  __shared__ int myRingBufferLoc; // tracks my TBs location in the ring buffer

  if (threadIdx.x == 0) {
    myRingBufferLoc = -1; // initially I have no location
  }
  __syncthreads();

  for (int i = 0; i < ITERATIONS; ++i)
  {
    myRingBufferLoc = cudaMutexSleepLockLocal(mutex, smID, gpuLockData->mutexBuffers,
                                              gpuLockData->mutexBufferTails, gpuLockData->maxBufferSize,
                                              gpuLockData->arrayStride, NUM_SM);

    accessData(storage, threadBaseLoc, threadOffset, NUM_LDST);

    cudaMutexSleepUnlockLocal(mutex, smID, gpuLockData->mutexBuffers, myRingBufferLoc,
                              gpuLockData->maxBufferSize, gpuLockData->arrayStride,
                              NUM_SM);
  }
}

__global__ void kernelFetchAndAddMutex(cudaMutex_t mutex, float * storage,
                                       cudaLockData_t * gpuLockData,
                                       const int ITERATIONS,
                                       const int NUM_LDST, const int NUM_SM)
{
  // local variables
  // all thread blocks access the same locations (rely on release to get
  // ownership in time)
  const int tid = threadIdx.x;
  // want striding to happen across cache lines so that each thread in a
  // half-warp accesses sequential words
  const int threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  const int threadOffset = (threadIdx.x % NUM_WORDS_PER_CACHELINE);

  for (int i = 0; i < ITERATIONS; ++i)
  {
    cudaMutexFALock(mutex, gpuLockData->mutexBufferHeads, gpuLockData->mutexBufferTails,
                    NUM_SM);

    accessData(storage, threadBaseLoc, threadOffset, NUM_LDST);

    cudaMutexFAUnlock(mutex, gpuLockData->mutexBufferTails, NUM_SM);
  }
}

__global__ void kernelFetchAndAddMutexUniq(cudaMutex_t mutex, float * storage,
                                           cudaLockData_t * gpuLockData,
                                           const int ITERATIONS,
                                           const int NUM_LDST,
                                           const int NUM_SM)
{
  // local variables
  const int smID = (blockIdx.x % NUM_SM); // mod by # SMs to get SM ID
  // all thread blocks on the same SM access the same locations
  const int tid = ((smID * blockDim.x) + threadIdx.x);
  // want striding to happen across cache lines so that each thread in a
  // half-warp accesses sequential words
  const int threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  const int threadOffset = (threadIdx.x % NUM_WORDS_PER_CACHELINE);

  for (int i = 0; i < ITERATIONS; ++i)
  {
    cudaMutexFALockLocal(mutex, smID, gpuLockData->mutexBufferHeads,
                         gpuLockData->mutexBufferTails, NUM_SM);

    accessData(storage, threadBaseLoc, threadOffset, NUM_LDST);

    cudaMutexFAUnlockLocal(mutex, smID, gpuLockData->mutexBufferTails, NUM_SM);
  }
}

__global__ void kernelSpinLockMutex(cudaMutex_t mutex, float * storage,
                                    cudaLockData_t * gpuLockData,
                                    const int ITERATIONS, const int NUM_LDST,
                                    const int NUM_SM)
{
  // local variables
  // all thread blocks access the same locations (rely on release to get
  // ownership in time)
  const int tid = threadIdx.x;
  // want striding to happen across cache lines so that each thread in a
  // half-warp accesses sequential words
  const int threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  const int threadOffset = (threadIdx.x % NUM_WORDS_PER_CACHELINE);

  for (int i = 0; i < ITERATIONS; ++i)
  {
    cudaMutexSpinLock(mutex, gpuLockData->mutexBufferHeads, NUM_SM);

    accessData(storage, threadBaseLoc, threadOffset, NUM_LDST);

    cudaMutexSpinUnlock(mutex, gpuLockData->mutexBufferHeads, NUM_SM);
  }
}

__global__ void kernelSpinLockMutexUniq(cudaMutex_t mutex, float * storage,
                                        cudaLockData_t * gpuLockData,
                                        const int ITERATIONS,
                                        const int NUM_LDST, const int NUM_SM)
{
  // local variables
  const int smID = (blockIdx.x % NUM_SM); // mod by # SMs to get SM ID
  // all thread blocks on the same SM access the same locations
  const int tid = ((smID * blockDim.x) + threadIdx.x);
  // want striding to happen across cache lines so that each thread in a
  // half-warp accesses sequential words
  const int threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  const int threadOffset = (threadIdx.x % NUM_WORDS_PER_CACHELINE);

  for (int i = 0; i < ITERATIONS; ++i)
  {
    cudaMutexSpinLockLocal(mutex, smID, gpuLockData->mutexBufferHeads, NUM_SM);

    accessData(storage, threadBaseLoc, threadOffset, NUM_LDST);

    cudaMutexSpinUnlockLocal(mutex, smID, gpuLockData->mutexBufferHeads, NUM_SM);
  }
}

__global__ void kernelEBOMutex(cudaMutex_t mutex, float * storage,
                               cudaLockData_t * gpuLockData,
                               const int ITERATIONS, const int NUM_LDST,
                               const int NUM_SM)
{
  // local variables
  // all thread blocks access the same locations (rely on release to get
  // ownership in time)
  const int tid = threadIdx.x;
  // want striding to happen across cache lines so that each thread in a
  // half-warp accesses sequential words
  const int threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  const int threadOffset = (threadIdx.x % NUM_WORDS_PER_CACHELINE);

  for (int i = 0; i < ITERATIONS; ++i)
  {
    cudaMutexEBOLock(mutex, gpuLockData->mutexBufferHeads, NUM_SM);

    accessData(storage, threadBaseLoc, threadOffset, NUM_LDST);

    cudaMutexEBOUnlock(mutex, gpuLockData->mutexBufferHeads, NUM_SM);
  }
}

__global__ void kernelEBOMutexUniq(cudaMutex_t mutex, float * storage,
                                   cudaLockData_t * gpuLockData,
                                   const int ITERATIONS, const int NUM_LDST,
                                   const int NUM_SM)
{
  // local variables
  const int smID = (blockIdx.x % NUM_SM); // mod by # SMs to get SM ID
  // all thread blocks on the same SM access the same locations
  const int tid = ((smID * blockDim.x) + threadIdx.x);
  // want striding to happen across cache lines so that each thread in a
  // half-warp accesses sequential words
  const int threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  const int threadOffset = (threadIdx.x % NUM_WORDS_PER_CACHELINE);

  for (int i = 0; i < ITERATIONS; ++i)
  {
    cudaMutexEBOLockLocal(mutex, smID, gpuLockData->mutexBufferHeads, NUM_SM);

    accessData(storage, threadBaseLoc, threadOffset, NUM_LDST);

    cudaMutexEBOUnlockLocal(mutex, smID, gpuLockData->mutexBufferHeads, NUM_SM);
  }
}

// All TBs on all SMs access the same data with 1 writer per SM (and N-1)
// readers per SM.
__global__ void kernelSpinLockSemaphore(cudaSemaphore_t sem,
                                        float * storage,
                                        cudaLockData_t * gpuLockData,
                                        const unsigned int numStorageLocs,
                                        const int ITERATIONS,
                                        const int NUM_LDST,
                                        const int NUM_SM)
{
  // local variables
  const unsigned int maxSemCount =
    gpuLockData->semaphoreBuffers[((sem * 4 * NUM_SM) + (0 * 4))];
  const int smID = (blockIdx.x % NUM_SM); // mod by # SMs to get SM ID
  // If there are fewer TBs than # SMs, need to take into account for various
  // math below.  If TBs >= NUM_SM, use NUM_SM.
  const unsigned int numSM = ((gridDim.x < NUM_SM) ? gridDim.x : NUM_SM);
  // given the gridDim.x, we can figure out how many TBs are on our SM -- assume
  // all SMs have an identical number of TBs
  int numTBs_perSM = (gridDim.x / numSM);
  if (numTBs_perSM == 0) { ++numTBs_perSM; } // always have to have at least 1
  // number of threads on each TB
  //const int numThrs_perSM = (blockDim.x * numTBs_perSM);
  const int perSM_blockID = (blockIdx.x / numSM);
  // rotate which TB is the writer
  const bool isWriter = (perSM_blockID == (smID % numTBs_perSM));

  // all thread blocks on the same SM access unique locations except the writer,
  // which writes all of the locations that all of the TBs access
  //int currBlockID = blockIdx.x;
  // the (reader) TBs on each SM access unique locations but those same
  // locations are accessed by the reader TBs on all SMs
  int tid = ((perSM_blockID * blockDim.x) + threadIdx.x);
  // want striding to happen across cache lines so that each thread in a
  // half-warp accesses sequential words
  int threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  const int threadOffset = (threadIdx.x % NUM_WORDS_PER_CACHELINE);

  // dummy array to hold the loads done in the readers
  __shared__ volatile float dummyArray[NUM_THREADS_PER_BLOCK];

  for (int i = 0; i < ITERATIONS; ++i)
  {
    /*
      NOTE: There is a race here for entering the critical section.  Most
      importantly, it means that the at least one of the readers could win and
      thus the readers will read before the writer has had a chance to write
      the data.
    */
    cudaSemaphoreSpinWait(sem, isWriter, maxSemCount,
                          gpuLockData->semaphoreBuffers, NUM_SM);

    if (isWriter) { // TB 0 writes all the data that the TBs on this SM access
      for (int j = 0; j < numTBs_perSM; ++j) {
        accessData_semWr(storage, threadBaseLoc, threadOffset, NUM_LDST);

        /*
          Update the writer's "location" so it writes to the locations that the
          readers will access (due to RR scheduling the next TB on this SM is
          numSM TBs away).  Use loop counter because the non-unique version
          writes the same locations on all SMs.
        */
        tid = (((j+1) * blockDim.x) + threadIdx.x) % numStorageLocs;
        threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
      }
      // reset locations
      tid = ((perSM_blockID * blockDim.x) + threadIdx.x);
      threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
    } else { // rest of TBs on this SM read the data written by each SM's TB 0
      accessData_semRd(storage, dummyArray, threadBaseLoc, threadOffset,
                       NUM_LDST);
    }
    cudaSemaphoreSpinPost(sem, isWriter, maxSemCount,
                          gpuLockData->semaphoreBuffers, NUM_SM);
  }
}

__global__ void kernelSpinLockSemaphoreUniq(cudaSemaphore_t sem,
                                            float * storage,
                                            cudaLockData_t * gpuLockData,
                                            const int ITERATIONS,
                                            const int NUM_LDST,
                                            const int NUM_SM)
{
  // local variables
  const unsigned int maxSemCount =
      gpuLockData->semaphoreBuffers[((sem * 4 * NUM_SM) + (0 * 4))];
  // If there are fewer TBs than # SMs, need to take into account for various
  // math below.  If TBs >= NUM_SM, use NUM_SM.
  const unsigned int numSM = ((gridDim.x < NUM_SM) ? gridDim.x : NUM_SM);
  const int smID = (blockIdx.x % numSM); // mod by # SMs to get SM ID
  // given the gridDim.x, we can figure out how many TBs are on our SM -- assume
  // all SMs have an identical number of TBs
  int numTBs_perSM = (gridDim.x / numSM);
  if (numTBs_perSM == 0) { ++numTBs_perSM; } // always have to have at least 1
  const int perSM_blockID = (blockIdx.x / numSM);
  // rotate which TB is the writer
  const bool isWriter = (perSM_blockID == (smID % numTBs_perSM));

  // all thread blocks on the same SM access unique locations except the writer,
  // which writes all of the locations that all of the TBs access
  int currBlockID = blockIdx.x;
  int tid = ((currBlockID * blockDim.x) + threadIdx.x);
  // want striding to happen across cache lines so that each thread in a
  // half-warp accesses sequential words
  int threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  const int threadOffset = (threadIdx.x % NUM_WORDS_PER_CACHELINE);
  // dummy array to hold the loads done in the readers
  __shared__ volatile float dummyArray[NUM_THREADS_PER_BLOCK];

  for (int i = 0; i < ITERATIONS; ++i)
  {
    /*
      NOTE: There is a race here for entering the critical section.  Most
      importantly, it means that the at least one of the readers could win and
      thus the readers will read before the writer has had a chance to write
      the data.
    */
    cudaSemaphoreSpinWaitLocal(sem, smID, isWriter, maxSemCount,
                               gpuLockData->semaphoreBuffers, NUM_SM);

    if (isWriter) { // TB 0 writes all the data that the TBs on this SM access
      for (int j = 0; j < numTBs_perSM; ++j) {
        accessData_semWr(storage, threadBaseLoc, threadOffset, NUM_LDST);

        /*
          update the writer's "location" so it writes to the locations that the
          readers will access (due to RR scheduling the next TB on this SM is
          numSM TBs away and < gridDim.x).
        */
        currBlockID = (currBlockID + numSM) % gridDim.x;
        tid = ((currBlockID * blockDim.x) + threadIdx.x);
        threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
      }
      // reset locations
      currBlockID = blockIdx.x;
      tid = ((currBlockID * blockDim.x) + threadIdx.x);
      threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
    } else { // rest of TBs on this SM read the data written by each SM's TB 0
      accessData_semRd(storage, dummyArray, threadBaseLoc, threadOffset,
                       NUM_LDST);
    }
    cudaSemaphoreSpinPostLocal(sem, smID, isWriter, maxSemCount,
                               gpuLockData->semaphoreBuffers, NUM_SM);
  }
}

// All TBs on all SMs access the same data with 1 writer per SM (and N-1)
// readers per SM.
__global__ void kernelEBOSemaphore(cudaSemaphore_t sem, float * storage,
                                   cudaLockData_t * gpuLockData,
                                   const unsigned int numStorageLocs,
                                   const int ITERATIONS, const int NUM_LDST,
                                   const int NUM_SM)
{
  // local variables
  const unsigned int maxSemCount =
      gpuLockData->semaphoreBuffers[((sem * 4 * NUM_SM) + (0 * 4))];
  // If there are fewer TBs than # SMs, need to take into account for various
  // math below.  If TBs >= NUM_SM, use NUM_SM.
  const unsigned int numSM = ((gridDim.x < NUM_SM) ? gridDim.x : NUM_SM);
  const int smID = (blockIdx.x % NUM_SM); // mod by # SMs to get SM ID
  // given the gridDim.x, we can figure out how many TBs are on our SM -- assume
  // all SMs have an identical number of TBs
  int numTBs_perSM = (gridDim.x / numSM);
  if (numTBs_perSM == 0) { ++numTBs_perSM; } // always have to have at least 1
  // number of threads on each TB
  //const int numThrs_perSM = (blockDim.x * numTBs_perSM);
  const int perSM_blockID = (blockIdx.x / numSM);
  // rotate which TB is the writer
  const bool isWriter = (perSM_blockID == (smID % numTBs_perSM));

  // all thread blocks on the same SM access unique locations except the writer,
  // which writes all of the locations that all of the TBs access
  //int currBlockID = blockIdx.x;
  // the (reader) TBs on each SM access unique locations but those same
  // locations are accessed by the reader TBs on all SMs
  int tid = ((perSM_blockID * blockDim.x) + threadIdx.x);
  // want striding to happen across cache lines so that each thread in a
  // half-warp accesses sequential words
  int threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  const int threadOffset = (threadIdx.x % NUM_WORDS_PER_CACHELINE);

  // dummy array to hold the loads done in the readers
  __shared__ volatile float dummyArray[NUM_THREADS_PER_BLOCK];

  for (int i = 0; i < ITERATIONS; ++i)
  {
    /*
      NOTE: There is a race here for entering the critical section.  Most
      importantly, it means that the at least one of the readers could win and
      thus the readers will read before the writer has had a chance to write
      the data.
    */
   cudaSemaphoreEBOWait(sem, isWriter, maxSemCount,
                        gpuLockData->semaphoreBuffers, NUM_SM);

    if (isWriter) { // TB 0 writes all the data that the TBs on this SM access
      for (int j = 0; j < numTBs_perSM; ++j) {
        accessData_semWr(storage, threadBaseLoc, threadOffset, NUM_LDST);

        /*
          Update the writer's "location" so it writes to the locations that the
          readers will access (due to RR scheduling the next TB on this SM is
          numSM TBs away).  Use loop counter because the non-unique version
          writes the same locations on all SMs.
        */
        tid = (((j+1) * blockDim.x) + threadIdx.x) % numStorageLocs;
        threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
      }
      // reset locations
      tid = ((perSM_blockID * blockDim.x) + threadIdx.x);
      threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
    } else { // rest of TBs on this SM read the data written by each SM's TB 0
      accessData_semRd(storage, dummyArray, threadBaseLoc, threadOffset,
                       NUM_LDST);
    }
    cudaSemaphoreEBOPost(sem, isWriter, maxSemCount,
                         gpuLockData->semaphoreBuffers, NUM_SM);
  }
}

__global__ void kernelEBOSemaphoreUniq(cudaSemaphore_t sem, float * storage,
                                       cudaLockData_t * gpuLockData,
                                       const int ITERATIONS,
                                       const int NUM_LDST,
                                       const int NUM_SM)
{
  // local variables
  const unsigned int maxSemCount =
      gpuLockData->semaphoreBuffers[((sem * 4 * NUM_SM) + (0 * 4))];
  // If there are fewer TBs than # SMs, need to take into account for various
  // math below.  If TBs >= NUM_SM, use NUM_SM.
  const unsigned int numSM = ((gridDim.x < NUM_SM) ? gridDim.x : NUM_SM);
  const int smID = (blockIdx.x % numSM); // mod by # SMs to get SM ID
  // given the gridDim.x, we can figure out how many TBs are on our SM -- assume
  // all SMs have an identical number of TBs
  int numTBs_perSM = (gridDim.x / numSM);
  if (numTBs_perSM == 0) { ++numTBs_perSM; } // always have to have at least 1
  const int perSM_blockID = (blockIdx.x / numSM);
  // rotate which TB is the writer
  const bool isWriter = (perSM_blockID == (smID % numTBs_perSM));

  // all thread blocks on the same SM access unique locations except the writer,
  // which writes all of the locations that all of the TBs access
  int currBlockID = blockIdx.x;
  int tid = ((currBlockID * blockDim.x) + threadIdx.x);
  // want striding to happen across cache lines so that each thread in a
  // half-warp accesses sequential words
  int threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
  const int threadOffset = (threadIdx.x % NUM_WORDS_PER_CACHELINE);
  // dummy array to hold the loads done in the readers
  __shared__ volatile float dummyArray[NUM_THREADS_PER_BLOCK];

  for (int i = 0; i < ITERATIONS; ++i)
  {
    /*
      NOTE: There is a race here for entering the critical section.  Most
      importantly, it means that the at least one of the readers could win and
      thus the readers will read before the writer has had a chance to write
      the data.
    */
    cudaSemaphoreEBOWaitLocal(sem, smID, isWriter, maxSemCount,
                              gpuLockData->semaphoreBuffers, NUM_SM);

    if (isWriter) { // TB 0 writes all the data that the TBs on this SM access
      for (int j = 0; j < numTBs_perSM; ++j) {
        accessData_semWr(storage, threadBaseLoc, threadOffset, NUM_LDST);

        /*
          update the writer's "location" so it writes to the locations that the
          readers will access (due to RR scheduling the next TB on this SM is
          numSM TBs away and < gridDim.x).
        */
        currBlockID = (currBlockID + numSM) % gridDim.x;
        tid = ((currBlockID * blockDim.x) + threadIdx.x);
        threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
      }
      // reset locations
      currBlockID = blockIdx.x;
      tid = ((currBlockID * blockDim.x) + threadIdx.x);
      threadBaseLoc = ((tid/NUM_WORDS_PER_CACHELINE) * (NUM_LDST+1));
    } else { // rest of TBs on this SM read the data written by each SM's TB 0
      accessData_semRd(storage, dummyArray, threadBaseLoc, threadOffset,
                       NUM_LDST);
    }
    cudaSemaphoreEBOPostLocal(sem, smID, isWriter, maxSemCount,
                              gpuLockData->semaphoreBuffers, NUM_SM);
  }
}

void invokeAtomicTreeBarrier(float * storage_d, unsigned int * perSMBarriers_d,
                             int numIters)
{
  // local variable
  const int blocks = numTBs;

  for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
  {
    kernelAtomicTreeBarrierUniq<<<blocks, NUM_THREADS_PER_BLOCK, 0, 0>>>(
        storage_d, cpuLockData, perSMBarriers_d, numIters, NUM_LDST, NUM_SM,
        MAX_BLOCKS);

    // Blocks until the device has completed all preceding requested
    // tasks (make sure that the device returned before continuing).
    cudaError_t cudaErr = cudaThreadSynchronize();
    checkError(cudaErr, "cudaThreadSynchronize (kernelAtomicTreeBarrierUniq)");
  }
}

void invokeAtomicTreeBarrierLocalExch(float * storage_d,
                                      unsigned int * perSMBarriers_d,
                                      int numIters)
{
  // local variable
  const int blocks = numTBs;

  for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
  {
    kernelAtomicTreeBarrierUniqLocalExch<<<blocks, NUM_THREADS_PER_BLOCK, 0, 0>>>(
        storage_d, cpuLockData, perSMBarriers_d, numIters, NUM_LDST, NUM_SM,
        MAX_BLOCKS);

    // Blocks until the device has completed all preceding requested
    // tasks (make sure that the device returned before continuing).
    cudaError_t cudaErr = cudaThreadSynchronize();
    checkError(cudaErr,
               "cudaThreadSynchronize (kernelAtomicTreeBarrierUniqLockExch)");
  }
}

void invokeFBSTreeBarrier(float * storage_d, unsigned int * perSMBarriers_d,
                          int numIters)
{
  // local variable
  const int blocks = numTBs;

  for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
  {
    kernelFBSTreeBarrierUniq<<<blocks, NUM_THREADS_PER_BLOCK, 0, 0>>>(
        storage_d, cpuLockData, perSMBarriers_d, numIters, NUM_LDST, NUM_SM,
        MAX_BLOCKS);

    // Blocks until the device has completed all preceding requested
    // tasks (make sure that the device returned before continuing).
    cudaError_t cudaErr = cudaThreadSynchronize();
    checkError(cudaErr, "cudaThreadSynchronize (kernelFBSTreeBarrierUniq)");
  }
}

void invokeFBSTreeBarrierLocalExch(float * storage_d,
                                   unsigned int * perSMBarriers_d,
                                   int numIters)
{
  // local variable
  const int blocks = numTBs;

  for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
  {
    kernelFBSTreeBarrierUniqLocalExch<<<blocks, NUM_THREADS_PER_BLOCK, 0, 0>>>(
        storage_d, cpuLockData, perSMBarriers_d, numIters, NUM_LDST, NUM_SM,
        MAX_BLOCKS);

    // Blocks until the device has completed all preceding requested
    // tasks (make sure that the device returned before continuing).
    cudaError_t cudaErr = cudaThreadSynchronize();
    checkError(cudaErr, "cudaThreadSynchronize (kernelFBSTreeBarrierUniqLocalExch)");
  }
}

void invokeSpinLockMutex(cudaMutex_t mutex, float * storage_d, int numIters)
{
  // local variable
  const int blocks = numTBs;

  for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
  {
    kernelSpinLockMutex<<<blocks, NUM_THREADS_PER_BLOCK>>>(
        mutex, storage_d, cpuLockData, numIters, NUM_LDST, NUM_SM);

    // Blocks until the device has completed all preceding requested
    // tasks (make sure that the device returned before continuing).
    cudaError_t cudaErr = cudaThreadSynchronize();
    checkError(cudaErr, "cudaThreadSynchronize (kernelSpinLockMutex)");
  }
}

void invokeSpinLockMutex_uniq(cudaMutex_t mutex, float * storage_d,
                              int numIters)
{
  // local variable
  const int blocks = numTBs;

  for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
  {
    kernelSpinLockMutexUniq<<<blocks, NUM_THREADS_PER_BLOCK, 0, 0>>>(
        mutex, storage_d, cpuLockData, numIters, NUM_LDST, NUM_SM);

    // Blocks until the device has completed all preceding requested
    // tasks (make sure that the device returned before continuing).
    cudaError_t cudaErr = cudaThreadSynchronize();
    checkError(cudaErr, "cudaThreadSynchronize (kernelSpinLockMutexUniq)");
  }
}

void invokeEBOMutex(cudaMutex_t mutex, float * storage_d, int numIters)
{
  // local variable
  const int blocks = numTBs;

  for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
  {
    kernelEBOMutex<<<blocks, NUM_THREADS_PER_BLOCK, 0, 0>>>(
        mutex, storage_d, cpuLockData, numIters, NUM_LDST, NUM_SM);

    // Blocks until the device has completed all preceding requested
    // tasks (make sure that the device returned before continuing).
    cudaError_t cudaErr = cudaThreadSynchronize();
    checkError(cudaErr, "cudaThreadSynchronize (kernelEBOMutex)");
  }
}

void invokeEBOMutex_uniq(cudaMutex_t mutex, float * storage_d, int numIters)
{
  // local variable
  const int blocks = numTBs;

  for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
  {
    kernelEBOMutexUniq<<<blocks, NUM_THREADS_PER_BLOCK, 0, 0>>>(
        mutex, storage_d, cpuLockData, numIters, NUM_LDST, NUM_SM);

    // Blocks until the device has completed all preceding requested
    // tasks (make sure that the device returned before continuing).
    cudaError_t cudaErr = cudaThreadSynchronize();
    checkError(cudaErr, "cudaThreadSynchronize (kernelEBOMutexUniq)");
  }
}

void invokeSleepingMutex(cudaMutex_t mutex, float * storage_d, int numIters)
{
  // local variable
  const int blocks = numTBs;

  for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
  {
    kernelSleepingMutex<<<blocks, NUM_THREADS_PER_BLOCK, 0, 0>>>(
        mutex, storage_d, cpuLockData, numIters, NUM_LDST, NUM_SM);

    // Blocks until the device has completed all preceding requested
    // tasks (make sure that the device returned before continuing).
    cudaError_t cudaErr = cudaThreadSynchronize();
    checkError(cudaErr, "cudaThreadSynchronize (kernelSleepingMutex)");
  }
}

void invokeSleepingMutex_uniq(cudaMutex_t mutex, float * storage_d,
                              int numIters)
{
  // local variable
  const int blocks = numTBs;

  for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
  {
    kernelSleepingMutexUniq<<<blocks, NUM_THREADS_PER_BLOCK, 0, 0>>>(
        mutex, storage_d, cpuLockData, numIters, NUM_LDST, NUM_SM);

    // Blocks until the device has completed all preceding requested
    // tasks (make sure that the device returned before continuing).
    cudaError_t cudaErr = cudaThreadSynchronize();
    checkError(cudaErr, "cudaThreadSynchronize (kernelSleepingMutexUniq)");
  }
}

void invokeFetchAndAddMutex(cudaMutex_t mutex, float * storage_d, int numIters)
{
  // local variable
  const int blocks = numTBs;

  for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
  {
    kernelFetchAndAddMutex<<<blocks, NUM_THREADS_PER_BLOCK, 0, 0>>>(
        mutex, storage_d, cpuLockData, numIters, NUM_LDST, NUM_SM);

    // Blocks until the device has completed all preceding requested
    // tasks (make sure that the device returned before continuing).
    cudaError_t cudaErr = cudaThreadSynchronize();
    checkError(cudaErr, "cudaThreadSynchronize (kernelFetchAndAddMutex)");
  }
}

void invokeFetchAndAddMutex_uniq(cudaMutex_t mutex, float * storage_d, int numIters)
{
  // local variable
  const int blocks = numTBs;

  for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
  {
    kernelFetchAndAddMutexUniq<<<blocks, NUM_THREADS_PER_BLOCK, 0, 0>>>(
        mutex, storage_d, cpuLockData, numIters, NUM_LDST, NUM_SM);

    // Blocks until the device has completed all preceding requested
    // tasks (make sure that the device returned before continuing).
    cudaError_t cudaErr = cudaThreadSynchronize();
    checkError(cudaErr, "cudaThreadSynchronize (kernelFetchAndAddMutexUniq)");
  }
}

void invokeSpinLockSemaphore(cudaSemaphore_t sem, float * storage_d,
                             const int maxVal,
                             int numIters, int numStorageLocs)
{
  // local variable
  const int blocks = numTBs;

  for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
  {
    kernelSpinLockSemaphore<<<blocks, NUM_THREADS_PER_BLOCK, 0, 0>>>(
        sem, storage_d, cpuLockData, numStorageLocs, numIters, NUM_LDST,
        NUM_SM);

    // Blocks until the device has completed all preceding requested
    // tasks (make sure that the device returned before continuing).
    cudaError_t cudaErr = cudaThreadSynchronize();
    checkError(cudaErr, "cudaThreadSynchronize (kernelSpinLockSemaphore)");
  }
}

void invokeSpinLockSemaphore_uniq(cudaSemaphore_t sem, float * storage_d,
                                  const int maxVal, int numIters)
{
  // local variable
  const int blocks = numTBs;

  for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
  {
    kernelSpinLockSemaphoreUniq<<<blocks, NUM_THREADS_PER_BLOCK, 0, 0>>>(
        sem, storage_d, cpuLockData, numIters, NUM_LDST, NUM_SM);

    // Blocks until the device has completed all preceding requested
    // tasks (make sure that the device returned before continuing).
    cudaError_t cudaErr = cudaThreadSynchronize();
    checkError(cudaErr, "cudaThreadSynchronize (kernelSpinLockSemaphoreUniq)");
  }
}

void invokeEBOSemaphore(cudaSemaphore_t sem, float * storage_d, const int maxVal,
                        int numIters, int numStorageLocs)
{
  // local variable
  const int blocks = numTBs;

  for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
  {
    kernelEBOSemaphore<<<blocks, NUM_THREADS_PER_BLOCK, 0, 0>>>(
        sem, storage_d, cpuLockData, numStorageLocs, numIters, NUM_LDST,
        NUM_SM);

    // Blocks until the device has completed all preceding requested
    // tasks (make sure that the device returned before continuing).
    cudaError_t cudaErr = cudaThreadSynchronize();
    checkError(cudaErr, "cudaThreadSynchronize (kernelEBOSemaphore)");
  }
}

void invokeEBOSemaphore_uniq(cudaSemaphore_t sem, float * storage_d, 
                             const int maxVal, int numIters)
{
  // local variable
  const int blocks = numTBs;

  for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
  {
    kernelEBOSemaphoreUniq<<<blocks, NUM_THREADS_PER_BLOCK, 0, 0>>>(
        sem, storage_d, cpuLockData, numIters, NUM_LDST, NUM_SM);
    
    // Blocks until the device has completed all preceding requested
    // tasks (make sure that the device returned before continuing).
    cudaError_t cudaErr = cudaThreadSynchronize();
    checkError(cudaErr, "cudaThreadSynchronize (kernelEBOSemaphoreUniq)");
  }
}

int main(int argc, char ** argv)
{
  if (argc != 5) {
    fprintf(stderr, "./allSyncPrims-1kernel <syncPrim> <numLdSt> <numTBs> "
            "<numCSIters>\n");
    fprintf(stderr, "where:\n");
    fprintf(stderr, "\t<syncPrim>: a string that represents which "
            "synchronization primitive to run.\n\t\tatomicTreeBarrUniq - "
            "Atomic Tree Barrier\n\t\tatomicTreeBarrUniqLocalExch - Atomic Tree "
            "Barrier with local exchange\n\t\tlfTreeBarrUniq - Lock-Free Tree "
            "Barrier\n\t\tlfTreeBarrUniqLocalExch - Lock-Free Tree Barrier with "
            "local exchange\n\t\tspinMutex - Spin Lock Mutex\n\t\tspinMutexEBO - Spin "
            "Lock Mutex with Backoff\n\t\tsleepMutex - Sleep Mutex\n\t\tfaMutex - "
            "Fetch-and-Add Mutex\n\t\tspinMutexUniq - Spin Lock Mutex -- accesses "
            "to unique locations per TB\n\t\tspinMutexEBOUniq - Spin Lock Mutex "
            "with Backoff -- accesses to unique locations per TB\n\t\t"
            "sleepMutexUniq - Sleep Mutex -- accesses to unique locations per "
            "TB\n\t\tfaMutexUniq - Fetch-and-Add Mutex -- accesses to unique "
            "locations per TB\n\t\tspinSemUniq1 - Spin Semaphore (Max: 1)\n\t\t"
            "spinSemUniq2 - Spin Semaphore (Max: 2)\n\t\tspinSemUniq10 - Spin "
            "Semaphore (Max: 10)\n\t\tspinSemUniq120 - Spin Semaphore (Max: 120)\n\t\t"
            "spinSemEBOUniq1 - Spin Semaphore with Backoff (Max: 1)\n\t\t"
            "spinSemEBOUniq2 - Spin Semaphore with Backoff (Max: 2)\n\t\t"
            "spinSemEBOUniq10 - Spin Semaphore with Backoff (Max: 10)\n\t\t"
            "spinSemEBOUniq120 - Spin Semaphore with Backoff (Max: 120)\n");
    fprintf(stderr, "\t<numLdSt>: the # of LDs and STs to do for each thread "
            "in the critical section.\n");
    fprintf(stderr, "\t<numTBs>: the # of TBs to execute (want to be "
            "divisible by the number of SMs).\n");
    fprintf(stderr, "\t<numCSIters>: number of iterations of the critical "
            "section.\n");
    exit(-1);
  }

  // boilerplate code to identify compute capability, # SM/SMM/SMX, etc.
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    fprintf(stderr, "There is no device supporting CUDA\n");
    exit(-1);
  }

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  fprintf(stdout, "GPU Compute Capability: %d.%d\n", deviceProp.major,
          deviceProp.minor);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {
    fprintf(stderr, "There is no CUDA capable device\n");
    exit(-1);
  }

  NUM_SM = deviceProp.multiProcessorCount;
  const int maxTBPerSM = deviceProp.maxThreadsPerBlock/NUM_THREADS_PER_BLOCK;
  //assert(maxTBPerSM * NUM_THREADS_PER_BLOCK <=
  //       deviceProp.maxThreadsPerMultiProcessor);
  MAX_BLOCKS = maxTBPerSM * NUM_SM;

  fprintf(stdout, "# SM: %d, Max Thrs/TB: %d, Max TB/SM: %d, Max # TB: %d\n",
          NUM_SM, deviceProp.maxThreadsPerBlock, maxTBPerSM, MAX_BLOCKS);

  cudaError_t cudaErr = cudaGetLastError();
  checkError(cudaErr, "Begin");

  // timing
  cudaEvent_t start, end;

  cudaErr = cudaEventCreate(&start);
  checkError(cudaErr, "cudaEventCreate (start)");
  cudaErr = cudaEventCreate(&end);
  checkError(cudaErr, "cudaEventCreate (end)");


  // parse input args
  const char * syncPrim_str = argv[1];
  NUM_LDST = atoi(argv[2]);
  numTBs = atoi(argv[3]);
  const int NUM_ITERS = atoi(argv[4]);
  assert(numTBs <= MAX_BLOCKS);
  const int numTBs_perSM = (int)ceil((float)numTBs / NUM_SM);

  unsigned int syncPrim = 9999;
  // set the syncPrim variable to the appropriate value based on the inputted
  // string for the microbenchmark
  if (strcmp(syncPrim_str, "atomicTreeBarrUniq") == 0) { syncPrim = 0; }
  else if (strcmp(syncPrim_str, "atomicTreeBarrUniqLocalExch") == 0) {
    syncPrim = 1;
  }
  else if (strcmp(syncPrim_str, "lfTreeBarrUniq") == 0) { syncPrim = 2; }
  else if (strcmp(syncPrim_str, "lfTreeBarrUniqLocalExch") == 0) {
    syncPrim = 3;
  }
  else if (strcmp(syncPrim_str, "spinMutex") == 0) { syncPrim = 4; }
  else if (strcmp(syncPrim_str, "spinMutexEBO") == 0) { syncPrim = 5; }
  else if (strcmp(syncPrim_str, "sleepMutex") == 0) { syncPrim = 6; }
  else if (strcmp(syncPrim_str, "faMutex") == 0) { syncPrim = 7; }
  else if (strcmp(syncPrim_str, "spinSem1") == 0) { syncPrim = 8; }
  else if (strcmp(syncPrim_str, "spinSem2") == 0) { syncPrim = 9; }
  else if (strcmp(syncPrim_str, "spinSem10") == 0) { syncPrim = 10; }
  else if (strcmp(syncPrim_str, "spinSem120") == 0) { syncPrim = 11; }
  else if (strcmp(syncPrim_str, "spinSemEBO1") == 0) { syncPrim = 12; }
  else if (strcmp(syncPrim_str, "spinSemEBO2") == 0) { syncPrim = 13; }
  else if (strcmp(syncPrim_str, "spinSemEBO10") == 0) { syncPrim = 14; }
  else if (strcmp(syncPrim_str, "spinSemEBO120") == 0) { syncPrim = 15; }
  // cases 16-19 reserved
  else if (strcmp(syncPrim_str, "spinMutexUniq") == 0) { syncPrim = 20; }
  else if (strcmp(syncPrim_str, "spinMutexEBOUniq") == 0) { syncPrim = 21; }
  else if (strcmp(syncPrim_str, "sleepMutexUniq") == 0) { syncPrim = 22; }
  else if (strcmp(syncPrim_str, "faMutexUniq") == 0) { syncPrim = 23; }
  else if (strcmp(syncPrim_str, "spinSemUniq1") == 0) { syncPrim = 24; }
  else if (strcmp(syncPrim_str, "spinSemUniq2") == 0) { syncPrim = 25; }
  else if (strcmp(syncPrim_str, "spinSemUniq10") == 0) { syncPrim = 26; }
  else if (strcmp(syncPrim_str, "spinSemUniq120") == 0) { syncPrim = 27; }
  else if (strcmp(syncPrim_str, "spinSemEBOUniq1") == 0) { syncPrim = 28; }
  else if (strcmp(syncPrim_str, "spinSemEBOUniq2") == 0) { syncPrim = 29; }
  else if (strcmp(syncPrim_str, "spinSemEBOUniq10") == 0) { syncPrim = 30; }
  else if (strcmp(syncPrim_str, "spinSemEBOUniq120") == 0) { syncPrim = 31; }
  // cases 32-36 reserved
  else
  {
    fprintf(stderr, "ERROR: Unknown synchronization primitive: %s\n",
            syncPrim_str);
    exit(-1);
  }

  // multiply number of mutexes, semaphores by NUM_SM to
  // allow per-core locks
  cudaLocksInit(MAX_BLOCKS, 8 * NUM_SM, 24 * NUM_SM, NUM_SM);

  cudaErr = cudaGetLastError();
  checkError(cudaErr, "After cudaLocksInit");

  /*
    The barriers need a per-SM barrier that is not part of the global synch
    structure.  In terms of size, for the lock-free barrier there are 2 arrays
    in here -- inVars and outVars.  Each needs to be sized to hold the maximum
    number of TBs/SM and each SM needs an array.

    The atomic barrier per-SM synchronization fits inside the lock-free size
    requirements so we can reuse the same locations.
  */
  unsigned int * perSMBarriers = (unsigned int *)malloc(sizeof(unsigned int) * (NUM_SM * MAX_BLOCKS * 2));

  int numLocsMult = 0;
  // barriers and unique semaphores have numTBs TBs accessing unique locations
  if ((syncPrim < 4) ||
      ((syncPrim >= 24) && (syncPrim <= 35))) { numLocsMult = numTBs; }
  // The non-unique mutex microbenchmarks, all TBs access the same locations so
  // multiplier is 1
  else if ((syncPrim >= 4) && (syncPrim <= 7)) { numLocsMult = 1; }
  // The non-unique semaphores have 1 writer and numTBs_perSM - 1 readers per SM
  // so the multiplier is numTBs_perSM
  else if ((syncPrim >= 8) && (syncPrim <= 19)) { numLocsMult = numTBs_perSM; }
  // For the unique mutex microbenchmarks and condition variable, all TBs on
  // same SM access same data so multiplier is NUM_SM.
  else if (((syncPrim >= 20) && (syncPrim <= 23)) ||
           (syncPrim == 36)) { numLocsMult = ((numTBs < NUM_SM) ?
                                              numTBs : NUM_SM); }
  else { // should never reach here
    fprintf(stderr, "ERROR: Unknown syncPrim: %u\n", syncPrim);
    exit(-1);
  }

  // each thread in a TB accesses NUM_LDST locations but accesses
  // per thread are offset so that each subsequent access is dependent
  // on the previous one -- thus need an extra access per thread.
  int numUniqLocsAccPerTB = (NUM_THREADS_PER_BLOCK * (NUM_LDST + 1));
  int numStorageLocs = (numLocsMult * numUniqLocsAccPerTB);
  float * storage = (float *)malloc(sizeof(float) * numStorageLocs);

  fprintf(stdout, "# TBs: %d, # Ld/St: %d, # Locs Mult: %d, # Uniq Locs/TB: %d, # Storage Locs: %d\n", numTBs, NUM_LDST, numLocsMult, numUniqLocsAccPerTB, numStorageLocs);

  // initialize storage
  for (int i = 0; i < numStorageLocs; ++i) { storage[i] = i; }
  // initialize per-SM barriers to 0's
  for (int i = 0; i < (NUM_SM * MAX_BLOCKS * 2); ++i) { perSMBarriers[i] = 0; }

  // gpu copies of storage and perSMBarriers
  unsigned int * perSMBarriers_d = NULL;
  float * storage_d = NULL;

  cudaMalloc(&perSMBarriers_d, sizeof(unsigned int) * (NUM_SM * MAX_BLOCKS * 2));
  cudaMalloc(&storage_d, sizeof(float) * numStorageLocs);

  cudaEventRecord(start, 0);

  cudaMemcpy(perSMBarriers_d, perSMBarriers, sizeof(unsigned int) * (NUM_SM * MAX_BLOCKS * 2), cudaMemcpyHostToDevice);
  cudaMemcpy(storage_d, storage, sizeof(float) * numStorageLocs, cudaMemcpyHostToDevice);

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  float elapsedTime = 0.0f;
  cudaEventElapsedTime(&elapsedTime, start, end);
  fprintf(stdout, "\tmemcpy H->D 2 elapsed time: %f ms\n", elapsedTime);
  fflush(stdout);

  // lock variables
  cudaMutex_t spinMutex, faMutex, sleepMutex, eboMutex;
  cudaMutex_t spinMutex_uniq, faMutex_uniq, sleepMutex_uniq, eboMutex_uniq;
  cudaSemaphore_t spinSem1, eboSem1,
                  spinSem2, eboSem2,
                  spinSem10, eboSem10,
                  spinSem120, eboSem120;
  cudaSemaphore_t spinSem1_uniq, eboSem1_uniq,
                  spinSem2_uniq, eboSem2_uniq,
                  spinSem10_uniq, eboSem10_uniq,
                  spinSem120_uniq, eboSem120_uniq;
  switch (syncPrim) {
    case 0: // atomic tree barrier doesn't require any special fields to be
            // created
      printf("atomic_tree_barrier_%03d\n", NUM_ITERS); fflush(stdout);
      break;
    case 1: // atomic tree barrier with local exchange doesn't require any
            // special fields to be created
      printf("atomic_tree_barrier_localExch_%03d\n", NUM_ITERS); fflush(stdout);
      break;
    case 2: // lock-free tree barrier doesn't require any special fields to be
            // created
      printf("fbs_tree_barrier_%03d\n", NUM_ITERS); fflush(stdout);
      break;
    case 3: // lock-free barrier with local exchange doesn't require any
            // special fields to be created
      printf("fbs_tree_barrier_localExch_%03d\n", NUM_ITERS); fflush(stdout);
      break;
    case 4:
      printf("spin_lock_mutex_%03d\n", NUM_ITERS); fflush(stdout);
      cudaMutexCreateSpin     (&spinMutex,          0);
      break;
    case 5:
      printf("ebo_mutex_%03d\n", NUM_ITERS); fflush(stdout);
      cudaMutexCreateEBO      (&eboMutex,           1);
      break;
    case 6:
      printf("sleeping_mutex_%03d\n", NUM_ITERS); fflush(stdout);
      cudaMutexCreateSleep    (&sleepMutex,         2);
      break;
    case 7:
      printf("fetchadd_mutex_%03d\n", NUM_ITERS); fflush(stdout);
      cudaMutexCreateFA       (&faMutex,            3);
      break;
    case 8:
      printf("spin_lock_sem_%03d_%03d\n", 1, NUM_ITERS); fflush(stdout);
      cudaSemaphoreCreateSpin (&spinSem1,      0,   1, NUM_SM);
      break;
    case 9:
      printf("spin_lock_sem_%03d_%03d\n", 2, NUM_ITERS); fflush(stdout);
      cudaSemaphoreCreateSpin (&spinSem2,      1,   2, NUM_SM);
      break;
    case 10:
      printf("spin_lock_sem_%03d_%03d\n", 10, NUM_ITERS); fflush(stdout);
      cudaSemaphoreCreateSpin (&spinSem10,      2,   10, NUM_SM);
      break;
    case 11:
      printf("spin_lock_sem_%03d_%03d\n", 2, NUM_ITERS); fflush(stdout);
      cudaSemaphoreCreateSpin (&spinSem120,      3,   120, NUM_SM);
      break;
    case 12:
      printf("ebo_sem_%03d_%03d\n", 1, NUM_ITERS); fflush(stdout);
      cudaSemaphoreCreateEBO  (&eboSem1,       4,   1, NUM_SM);
      break;
    case 13:
      printf("ebo_sem_%03d_%03d\n", 2, NUM_ITERS); fflush(stdout);
      cudaSemaphoreCreateEBO  (&eboSem2,       5,   2, NUM_SM);
      break;
    case 14:
      printf("ebo_sem_%03d_%03d\n", 10, NUM_ITERS); fflush(stdout);
      cudaSemaphoreCreateEBO  (&eboSem10,       6,   10, NUM_SM);
      break;
    case 15:
      printf("ebo_sem_%03d_%03d\n", 120, NUM_ITERS); fflush(stdout);
      cudaSemaphoreCreateEBO  (&eboSem120,       7,   120, NUM_SM);
      break;
    // cases 16-19 reserved
    case 16:
      break;
    case 17:
      break;
    case 18:
      break;
    case 19:
      break;
    case 20:
      printf("spin_lock_mutex_uniq_%03d\n", NUM_ITERS); fflush(stdout);
      cudaMutexCreateSpin     (&spinMutex_uniq,          4);
      break;
    case 21:
      printf("ebo_mutex_uniq_%03d\n", NUM_ITERS); fflush(stdout);
      cudaMutexCreateEBO      (&eboMutex_uniq,           5);
      break;
    case 22:
      printf("sleeping_mutex_uniq_%03d\n", NUM_ITERS); fflush(stdout);
      cudaMutexCreateSleep    (&sleepMutex_uniq,         6);
      break;
    case 23:
      printf("fetchadd_mutex_uniq_%03d\n", NUM_ITERS); fflush(stdout);
      cudaMutexCreateFA       (&faMutex_uniq,            7);
      break;
    case 24:
      printf("spin_lock_sem_uniq_%03d_%03d\n", 1, NUM_ITERS); fflush(stdout);
      cudaSemaphoreCreateSpin (&spinSem1_uniq,      12,   1, NUM_SM);
      break;
    case 25:
      printf("spin_lock_sem_uniq_%03d_%03d\n", 2, NUM_ITERS); fflush(stdout);
      cudaSemaphoreCreateSpin (&spinSem2_uniq,      13,   2, NUM_SM);
      break;
    case 26:
      printf("spin_lock_sem_uniq_%03d_%03d\n", 10, NUM_ITERS); fflush(stdout);
      cudaSemaphoreCreateSpin (&spinSem10_uniq,      14,   10, NUM_SM);
      break;
    case 27:
      printf("spin_lock_sem_uniq_%03d_%03d\n", 2, NUM_ITERS); fflush(stdout);
      cudaSemaphoreCreateSpin (&spinSem120_uniq,      15,   120, NUM_SM);
      break;
    case 28:
      printf("ebo_sem_uniq_%03d_%03d\n", 1, NUM_ITERS); fflush(stdout);
      cudaSemaphoreCreateEBO  (&eboSem1_uniq,       16,   1, NUM_SM);
      break;
    case 29:
      printf("ebo_sem_uniq_%03d_%03d\n", 2, NUM_ITERS); fflush(stdout);
      cudaSemaphoreCreateEBO  (&eboSem2_uniq,       17,   2, NUM_SM);
      break;
    case 30:
      printf("ebo_sem_uniq_%03d_%03d\n", 10, NUM_ITERS); fflush(stdout);
      cudaSemaphoreCreateEBO  (&eboSem10_uniq,       18,   10, NUM_SM);
      break;
    case 31:
      printf("ebo_sem_uniq_%03d_%03d\n", 120, NUM_ITERS); fflush(stdout);
      cudaSemaphoreCreateEBO  (&eboSem120_uniq,       19,   120, NUM_SM);
      break;
    // cases 32-36 reserved
    case 32:
      break;
    case 33:
      break;
    case 34:
      break;
    case 35:
      break;
    case 36:
      break;
    default:
      fprintf(stderr, "ERROR: Trying to run synch prim #%u, but only 0-36 are "
              "supported\n", syncPrim);
      exit(-1);
      break;
  }

  // # TBs must be < maxBufferSize or sleep mutex ring buffer won't work
  if ((syncPrim == 6) || (syncPrim == 22)) {
    assert(MAX_BLOCKS <= cpuLockData->maxBufferSize);
  }

  // NOTE: region of interest begins here
  cudaThreadSynchronize();
  cudaEventRecord(start, 0);

  switch (syncPrim) {
    case 0: // atomic tree barrier
      invokeAtomicTreeBarrier(storage_d, perSMBarriers_d, NUM_ITERS);
      break;
    case 1: // atomic tree barrier with local exchange
      invokeAtomicTreeBarrierLocalExch(storage_d, perSMBarriers_d, NUM_ITERS);
      break;
    case 2: // lock-free barrier
      invokeFBSTreeBarrier(storage_d, perSMBarriers_d, NUM_ITERS);
      break;
    case 3: // lock-free barrier with local exchange
      invokeFBSTreeBarrierLocalExch(storage_d, perSMBarriers_d, NUM_ITERS);
      break;
    case 4: // Spin Lock Mutex
      invokeSpinLockMutex   (spinMutex,  storage_d, NUM_ITERS);
      break;
    case 5: // Spin Lock Mutex with backoff
      invokeEBOMutex        (eboMutex,   storage_d, NUM_ITERS);
      break;
    case 6: // Sleeping Mutex
      invokeSleepingMutex   (sleepMutex, storage_d, NUM_ITERS);
      break;
    case 7: // fetch-and-add mutex
      invokeFetchAndAddMutex(faMutex,    storage_d, NUM_ITERS);
      break;
    case 8: // spin semaphore (1)
      invokeSpinLockSemaphore(spinSem1,   storage_d,   1, NUM_ITERS, numStorageLocs);
      break;
    case 9: // spin semaphore (2)
      invokeSpinLockSemaphore(spinSem2,   storage_d,   2, NUM_ITERS, numStorageLocs);
      break;
    case 10: // spin semaphore (10)
      invokeSpinLockSemaphore(spinSem10,   storage_d,   10, NUM_ITERS, numStorageLocs);
      break;
    case 11: // spin semaphore (120)
      invokeSpinLockSemaphore(spinSem120,   storage_d,   120, NUM_ITERS, numStorageLocs);
      break;
    case 12: // spin semaphore with backoff (1)
      invokeEBOSemaphore(eboSem1,   storage_d,     1, NUM_ITERS, numStorageLocs);
      break;
    case 13: // spin semaphore with backoff (2)
      invokeEBOSemaphore(eboSem2,   storage_d,     2, NUM_ITERS, numStorageLocs);
      break;
    case 14: // spin semaphore with backoff (10)
      invokeEBOSemaphore(eboSem10,   storage_d,   10, NUM_ITERS, numStorageLocs);
      break;
    case 15: // spin semaphore with backoff (120)
      invokeEBOSemaphore(eboSem120,   storage_d, 120, NUM_ITERS, numStorageLocs);
      break;
    // cases 16-19 reserved
    case 16:
      break;
    case 17:
      break;
    case 18:
      break;
    case 19:
      break;
    case 20: // Spin Lock Mutex (uniq)
      invokeSpinLockMutex_uniq   (spinMutex_uniq,  storage_d, NUM_ITERS);
      break;
    case 21: // Spin Lock Mutex with backoff (uniq)
      invokeEBOMutex_uniq        (eboMutex_uniq,   storage_d, NUM_ITERS);
      break;
    case 22: // Sleeping Mutex (uniq)
      invokeSleepingMutex_uniq   (sleepMutex_uniq, storage_d, NUM_ITERS);
      break;
    case 23: // fetch-and-add mutex (uniq)
      invokeFetchAndAddMutex_uniq(faMutex_uniq,    storage_d, NUM_ITERS);
      break;
    case 24: // spin semaphore (1) (uniq)
      invokeSpinLockSemaphore_uniq(spinSem1_uniq,   storage_d,   1, NUM_ITERS);
      break;
    case 25: // spin semaphore (2) (uniq)
      invokeSpinLockSemaphore_uniq(spinSem2_uniq,   storage_d,   2, NUM_ITERS);
      break;
    case 26: // spin semaphore (10) (uniq)
      invokeSpinLockSemaphore_uniq(spinSem10_uniq,   storage_d,   10, NUM_ITERS);
      break;
    case 27: // spin semaphore (120) (uniq)
      invokeSpinLockSemaphore_uniq(spinSem120_uniq,   storage_d,   120, NUM_ITERS);
      break;
    case 28: // spin semaphore with backoff (1) (uniq)
      invokeEBOSemaphore_uniq(eboSem1_uniq,   storage_d,     1, NUM_ITERS);
      break;
    case 29: // spin semaphore with backoff (2) (uniq)
      invokeEBOSemaphore_uniq(eboSem2_uniq,   storage_d,     2, NUM_ITERS);
      break;
    case 30: // spin semaphore with backoff (10) (uniq)
      invokeEBOSemaphore_uniq(eboSem10_uniq,   storage_d,   10, NUM_ITERS);
      break;
    case 31: // spin semaphore with backoff (120) (uniq)
      invokeEBOSemaphore_uniq(eboSem120_uniq,   storage_d, 120, NUM_ITERS);
      break;
    // cases 32-36 reserved
    case 32:
      break;
    case 33:
      break;
    case 34:
      break;
    case 35:
      break;
    case 36:
      break;
    default:
      fprintf(stderr,
              "ERROR: Trying to run synch prim #%u, but only 0-36 are "
              "supported\n",
              syncPrim);
      exit(-1);
      break;
  }

  // NOTE: Can end simulation here if don't care about output checking
  cudaThreadSynchronize();
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsedTime, start, end);
  float aveElapsedTime = elapsedTime/(float)NUM_REPEATS;
  printf("elapsed time: %f ms, average: %f ms\n", elapsedTime, aveElapsedTime);
  fflush(stdout);

  cudaEventRecord(start, 0);

  // copy results back to compare to golden
  cudaMemcpy(storage, storage_d, sizeof(float) * numStorageLocs, cudaMemcpyDeviceToHost);

  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsedTime, start, end);
  fprintf(stdout, "\tmemcpy D->H elapsed time: %f ms\n", elapsedTime);
  fflush(stdout);

  // get golden results
  float storageGolden[numStorageLocs];
  int numLocsAccessed = 0, currLoc = 0;
  // initialize
  for (int i = 0; i < numStorageLocs; ++i) { storageGolden[i] = i; }

  for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
  {
    for (int j = 0; j < NUM_ITERS; ++j)
    {
      /*
        The barrier algorithms exchange data across SMs, so we need to perform
        the exchanges in the golden code.

        The barrier algorithms with local exchange exchange data both across
        SMs and across TBs within an SM, so need to perform both in the golden
        code.
      */
      if (syncPrim < 4)
      {
        // Some kernels only access a fraction of the total # of locations,
        // determine how many locations are accessed by each kernel here.
        numLocsAccessed = (numTBs * numUniqLocsAccPerTB);

        // first cache line of words aren't written to
        for (int i = (numLocsAccessed-1); i >= 0; --i)
        {
          // every iteration of the critical section, the location being
          // accessed is shifted by numUniqLocsAccPerTB
          currLoc = (i + (j * numUniqLocsAccPerTB)) % numLocsAccessed;

          accessData_golden(storageGolden, currLoc, numStorageLocs);
        }

        // local exchange versions do additional accesses when there are
        // multiple TBs on a SM
        if ((syncPrim == 1) || (syncPrim == 3))
        {
          if (numTBs_perSM > 1)
          {
            for (int i = (numLocsAccessed-1); i >= 0; --i)
            {
              // advance the current location by the number of unique locations
              // accessed by a SM (mod the number of memory locations accessed)
              currLoc = (i + (numTBs_perSM * numUniqLocsAccPerTB)) %
                        numLocsAccessed;
              // every iteration of the critical section, the location being
              // accessed is also shifted by numUniqLocsAccPerTB
              currLoc = (currLoc + (j * numUniqLocsAccPerTB)) %
                        numLocsAccessed;

              accessData_golden(storageGolden, currLoc, numStorageLocs);
            }
          }
        }
      }
      /*
        In the non-unique microbenchmarks (4-19), all TBs on all SMs access
        the same locations

        ** NOTE: The semaphores do a reader-writer format but only the writer
        actually writes these locations, so this checking should get the
        right answers too.
      */
      else if ((syncPrim >= 4) && (syncPrim <= 19))
      {
        // need to iterate over the locations for each block since all TBs
        // access the same locations
        for (int block = 0; block < numTBs; ++block)
        {
          for (int i = (numUniqLocsAccPerTB-1); i >= 0; --i)
          {
            accessData_golden(storageGolden, i, numStorageLocs);
          }
        }
      }
      /*
        In the unique microbenchmarks (20-36), all TBs on an SM access
        the same data and the data accessed by each SM is unique.

        ** NOTE: The semaphores do a reader-writer format but only the writer
        actually writes these locations, so this checking should get the right
        answers too.
      */
      else
      {
        // Some kernels only access a fraction of the total # of locations,
        // determine how many locations are accessed by each kernel here.
        numLocsAccessed = (numTBs * numUniqLocsAccPerTB);
        // first cache line of words aren't written to
        for (int i = (numLocsAccessed-1); i >= 0; --i)
        {
          /*
            If this location would be accessed by a TB other than the first
            TB on an SM, wraparound and access the same location as the
            first TB on the SM -- only for the mutexes, for semaphores this
            isn't true.
          */
          currLoc = (((syncPrim >= 24) && (syncPrim <= 35)) ? i :
                     (i % (NUM_SM * numUniqLocsAccPerTB)));

          accessData_golden(storageGolden, currLoc, numStorageLocs);
        }
      }
    }
  }

  fprintf(stdout, "Comparing GPU results to golden results:\n");
  unsigned int numErrors = 0;
  // check the output values
  for (int i = 0; i < numStorageLocs; ++i)
  {
    if (std::abs(storage[i] - storageGolden[i]) > 1E-5)
    {
      fprintf(stderr, "\tERROR: storage[%d] = %f, golden[%d] = %f\n", i,
              storage[i], i, storageGolden[i]);
      ++numErrors;
    }
  }
  if (numErrors > 0)
  {
    fprintf(stderr, "ERROR: %s has %u output errors\n", syncPrim_str,
            numErrors);
    exit(-1);
  }
  else { fprintf(stdout, "PASSED!\n"); }

  // free timers and arrays
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  cudaLocksDestroy();
  cudaFree(storage_d);
  cudaFree(perSMBarriers_d);
  free(storage);
  free(perSMBarriers);

  return 0;
}
