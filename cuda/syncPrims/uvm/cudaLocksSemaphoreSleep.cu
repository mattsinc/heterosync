#ifndef __CUDASEMAPHORESLEEP_CU__
#define __CUDASEMAPHORESLEEP_CU__

inline __host__ cudaError_t cudaSemaphoreCreateSleep(cudaSemaphore_t * const handle,
                                                     const int semaphoreNumber,
                                                     const unsigned count,
                                                     const int NUM_SM)
{
  // In order: current count, head, tail, maximum count.
  unsigned int array[4] = { 0, 0, 0, count };
  for (int id = 0; id < NUM_SM; ++id) { // need to set these values for all SMs
    cudaMemcpy(&(cpuLockData->semaphoreBuffers[((semaphoreNumber * 4 * NUM_SM) + (id * 4))]), &(array[0]), sizeof(array[0]), cudaMemcpyHostToDevice);
    cudaMemcpy(&(cpuLockData->semaphoreBuffers[((semaphoreNumber * 4 * NUM_SM) + (id * 4)) + 1]), &(array[1]), sizeof(array[1]), cudaMemcpyHostToDevice);
    cudaMemcpy(&(cpuLockData->semaphoreBuffers[((semaphoreNumber * 4 * NUM_SM) + (id * 4)) + 2]), &(array[2]), sizeof(array[2]), cudaMemcpyHostToDevice);
    cudaMemcpy(&(cpuLockData->semaphoreBuffers[((semaphoreNumber * 4 * NUM_SM) + (id * 4)) + 3]), &(array[3]), sizeof(array[3]), cudaMemcpyHostToDevice);
  }
  *handle = semaphoreNumber;
  return cudaSuccess;
}

//  My idea for semaphores.
//
//  wait:
//
//    increment sem
//    if sem >= count
//      old pos = atomicInc(tail)
//      busy wait until head >= old pos
//    else
//      just go
//
//  post:
//    decrement sem
//    if old value > count
//      atomicInc(head)
inline __device__ void cudaSemaphoreSleepWait(const cudaSemaphore_t sem,
                                              const bool isWriter,
                                              const unsigned int maxCount,
                                              unsigned int * semaphoreBuffers,
                                              const int NUM_SM)
{
  const bool isMasterThread = (threadIdx.x == 0 && threadIdx.y == 0 &&
                               threadIdx.z == 0);
  // Each sem has NUM_SM * 4 locations in the buffer.  Of these locations, each
  // SM uses 4 of them (current count, head, tail, max count).  So SM 0 starts
  // at semaphoreBuffers[sem * 4 * NUM_SM].
  unsigned int * const currCount = semaphoreBuffers + (sem * 4 * NUM_SM);
  unsigned int * const semTail   = currCount + 2;
  __shared__ unsigned int waitIndex;
  __shared__ bool done;

  if (isMasterThread)
  {
    unsigned int countValue = 0;

    // because we write to the count without checking with the max count is, the
    // current count can exceed the max count, so need to handle appropriately
    if (isWriter) {
      // writers add maxCount instead of 1
      countValue = atomicAdd(currCount, maxCount);
      // adjusts for wraparound (theoretically will never happen)
      if (countValue >= 1000000000) {
        // atomicSub for now because the function writes are not working
        atomicSub(currCount, countValue); // set the count to 0
      }

      // writer can only enter if no readers are in the critical section
      if (countValue == maxCount) {
        waitIndex = 0;
        done = true;
      } else { // writer waits for its ticket to come up
        waitIndex = atomicInc(semTail, 1000000000);
        done = false;
      }
    }
    /*
      for readers: if the new, incremented count is still within the acceptable
      range, I can enter critical section right away, else note my place in line
      with the tail and wait for my "ticket" to come up.
    */
    else {
      // increment the number of TBs in the semaphore
      countValue = atomicInc(currCount, 1000000000);

      if (countValue < maxCount) {
        waitIndex = 0;
        done = true;
      } else {
        waitIndex = atomicInc(semTail, 1000000000);
        done = false;
      }
    }
  }
  __syncthreads();

  while (!done) // spin here waiting for space in the semaphore to open up
  {
    __syncthreads();
    if (isMasterThread)
    {
      const unsigned int head = ((volatile unsigned int *)semaphoreBuffers)[sem * 4 * NUM_SM + 1];

      // for the writer it's not good enough for its ticket to come up.  It also
      // needs to ensure that all previous readers are done
      if (isWriter) {
        if (head > waitIndex) {
          if (atomicAdd(currCount, 0) == 0) {
            // volatile read acts as a load acquire, need TF to enforce ordering locally
            __threadfence_block();
            done = true;
          }
          else                              { done = false; }
        } else                              { done = false; }
      } else {
        if (head > waitIndex)               {
          // volatile read acts as a load acquire, need TF to enforce ordering locally
          __threadfence_block();
          done = true;
        }
        else                                { done = false; }
      }
    }
    __syncthreads();
  }
  __syncthreads();
}

inline __device__ void cudaSemaphoreSleepPost(const cudaSemaphore_t sem,
                                              const bool isWriter,
                                              const unsigned int maxCount,
                                              unsigned int * semaphoreBuffers,
                                              const int NUM_SM)
{
  bool isMasterThread = (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0);

  if (isMasterThread)
  {
    unsigned int * const currCount   = semaphoreBuffers + (sem * 4 * NUM_SM);
    unsigned int * const semHead     = currCount + 1; // The "post" counter
    unsigned int oldCount = 0;

    // writer needs to decrement the count by maxCount since it took up the
    // entire semaphore
    if (isWriter) {
      oldCount = atomicSub(currCount, maxCount);

      // make sure count didn't go below zero
      if (oldCount < 0) {
        // ** NOTE: Since there is no lock on currCount this isn't really
        // safe but the count is absolute so it should be ok to add back
        // in the overages we incurred
        atomicAdd(currCount, (0-oldCount));
      }
    } else { // readers decrement count by 1
      oldCount = atomicDec(currCount, 1000000000);
    }

    if (oldCount > maxCount)
    {
      // atomicExch acts as a store release, need TF to enforce ordering
      __threadfence();
      atomicInc(semHead, 1000000000); // Arbitrarily high value.
    }
  }
  __syncthreads();
}

// same algorithm but with per-SM synchronization
inline __device__ void cudaSemaphoreSleepWaitLocal(const cudaSemaphore_t sem,
                                                   const unsigned int smID,
                                                   const bool isWriter,
                                                   const unsigned int maxCount,
                                                   unsigned int * semaphoreBuffers,
                                                   const int NUM_SM)
{
  const bool isMasterThread = (threadIdx.x == 0 && threadIdx.y == 0 &&
                               threadIdx.z == 0);
  /*
    Each sem has NUM_SM * 4 locations in the buffer.  Of these locations, each
    SM uses 4 of them (current count, head, tail, max count).  So SM 0 starts
    at semaphoreBuffers[sem * 4 * NUM_SM].
  */
  unsigned int * const currCount   = semaphoreBuffers + ((sem * 4 * NUM_SM) + (smID * 4));
  unsigned int * const semTail     = currCount + 2;
  __shared__ unsigned int waitIndex;
  __shared__ bool done;

  if (isMasterThread)
  {
    unsigned int countValue = 0;

    // because we write to the count without checking with the max count is, the
    // current count can exceed the max count, so need to handle appropriately
    if (isWriter) {
      // writers add maxCount instead of 1
      countValue = atomicAdd(currCount, maxCount);
      // adjusts for wraparound (theoretically will never happen)
      if (countValue >= 1000000000) {
        // atomicSub for now because the function writes are not working
        atomicSub(currCount, countValue); // set the count to 0
      }

      // writer can only enter if no readers are in the critical section
      if (countValue == maxCount) {
        waitIndex = 0;
        done = true;
      } else { // writer waits for its ticket to come up otherwise
        waitIndex = atomicInc(semTail, 1000000000);
        done = false;
      }
    }
    /*
      for readers: if the new, incremented count is still within the acceptable
      range, I can enter critical section right away, else note my place in line
      with the tail and wait for my "ticket" to come up.
    */
    else {
      // increment the number of TBs in the semaphore
      countValue = atomicInc(currCount, 1000000000);

      if (countValue < maxCount)
      {
        waitIndex = 0;
        done = true;
      }
      else
      {
        waitIndex = atomicInc(semTail, 1000000000);
        done = false;
      }
    }
  }
  __syncthreads();

  while (!done) // spin here waiting for space in the semaphore to open up
  {
    __syncthreads();
    if (isMasterThread)
    {
      const unsigned int head = ((volatile unsigned int *)semaphoreBuffers)[((sem * 4 * NUM_SM) + (smID * 4)) + 1];

      // for the writer it's not good enough for its ticket to come up.  It also
      // needs to ensure that all previous readers are done
      if (isWriter) {
        if (head > waitIndex) {
          if (atomicAdd(currCount, 0) == 0) {
            // volatile read acts as a load acquire, need TF to enforce ordering locally
            __threadfence_block();
            done = true;
          }
          else                              { done = false; }
        } else                              { done = false; }
      } else {
        if (head > waitIndex)               {
          // volatile read acts as a load acquire, need TF to enforce ordering locally
          __threadfence_block();
          done = true;
        }
        else                                { done = false; }
      }
    }
    __syncthreads();
  }
  __syncthreads();
}

// same algorithm but with per-SM synchronization
inline __device__ void cudaSemaphoreSleepPostLocal(const cudaSemaphore_t sem,
                                                   const unsigned int smID,
                                                   const bool isWriter,
                                                   const unsigned int maxCount,
                                                   unsigned int * semaphoreBuffers,
                                                   const int NUM_SM)
{
  bool isMasterThread = (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0);

  if (isMasterThread)
  {
    unsigned int * const currCount   = semaphoreBuffers + ((sem * 4 * NUM_SM) + (smID * 4));
    unsigned int * const semHead     = currCount + 1; // The "post" counter
    unsigned int oldCount = 0;

    // writer needs to decrement the count by maxCount since it took up the
    // entire semaphore
    if (isWriter) {
      oldCount = atomicSub(currCount, maxCount);

      // make sure count didn't go below zero
      if (oldCount < 0) {
        // ** NOTE: Since there is no lock on currCount this isn't really
        // safe but the count is absolute so it should be ok to add back
        // in the overages we incurred
        atomicAdd(currCount, (0-oldCount));
      }
    } else { // readers decrement count by 1
      oldCount = atomicDec(currCount, 1000000000);
    }

    if (oldCount > maxCount) {
      // atomicInc acts as a store release, need TF to enforce ordering locally
      __threadfence_block();
      atomicInc(semHead, 1000000000); // Arbitrarily high value.
    }
  }
  __syncthreads();
}

#endif
