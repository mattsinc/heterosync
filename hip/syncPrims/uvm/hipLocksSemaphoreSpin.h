#ifndef __HIPLOCKSSEMAPHORESPIN_H__
#define __HIPLOCKSSEMAPHORESPIN_H__

#include "hip/hip_runtime.h"
#include "hipLocks.h"

inline __host__ hipError_t hipSemaphoreCreateSpin(hipSemaphore_t * const handle,
                                                  const int semaphoreNumber,
                                                  const unsigned int count,
                                                  const int NUM_CU)
{
  // Here we set the initial value to be count+1, this allows us to do an
  // atomicExch(sem, 0) and basically use the semaphore value as both a
  // lock and a semaphore.
  unsigned int initialValue = (count + 1), zero = 0, one = 1;
  *handle = semaphoreNumber;
  for (int id = 0; id < NUM_CU; ++id) { // need to set these values for all CUs
    // Current count of the semaphore hence initialized to count+1
    hipMemcpy(&(cpuLockData->semaphoreBuffers[((semaphoreNumber * 5 * NUM_CU) + (id * 5))]), &initialValue, sizeof(initialValue), hipMemcpyHostToDevice);
    // Lock variable initialized to 0
    hipMemcpy(&(cpuLockData->semaphoreBuffers[((semaphoreNumber * 5 * NUM_CU) + (id * 5)) + 1]), &zero, sizeof(zero), hipMemcpyHostToDevice);
    // Writer waiting flag initialized to 1 to force writer to go first
    hipMemcpy(&(cpuLockData->semaphoreBuffers[((semaphoreNumber * 5 * NUM_CU) + (id * 5)) + 2]), &one, sizeof(one), hipMemcpyHostToDevice);
    // Max count for the semaphore hence initialized it to count+1
    hipMemcpy(&(cpuLockData->semaphoreBuffers[((semaphoreNumber * 5 * NUM_CU) + (id * 5)) + 3]), &initialValue, sizeof(initialValue), hipMemcpyHostToDevice);
    // Priority count initialized to 0
    hipMemcpy(&(cpuLockData->semaphoreBuffers[((semaphoreNumber * 5 * NUM_CU) + (id * 5)) + 4]), &zero, sizeof(zero), hipMemcpyHostToDevice);
  }
  return hipSuccess;
}

inline __device__ bool hipSemaphoreSpinTryWait(const hipSemaphore_t sem,
                                               const bool isWriter,
                                               const unsigned int maxSemCount,
                                               unsigned int * semaphoreBuffers,
                                               const int NUM_CU)
{
  const bool isMasterThread = (threadIdx.x == 0 && threadIdx.y == 0 &&
                               threadIdx.z == 0);
  /*
    Each sem has NUM_CU * 4 locations in the buffer.  Of these locations, each
    CU uses 4 of them (current count, head, tail, max count).  For the global
    semaphore all CUs use semaphoreBuffers[sem * 4 * NUM_CU].
  */
  unsigned int * const currCount = semaphoreBuffers + (sem * 4 * NUM_CU);
  unsigned int * const lock = currCount + 1;
  /*
    Reuse the tail for the "writers are waiting" flag since tail is unused.

    For now just use to indicate that at least 1 writer is waiting instead of
    a count to make sure that readers aren't totally starved out until all the
    writers are done.
  */
  unsigned int * const writerWaiting = currCount + 2;
  __shared__ bool acq1, acq2;

  __syncthreads();
  if (isMasterThread)
  {
    acq1 = false;
    // try to acquire the sem head "lock"
    if (atomicCAS(lock, 0, 1) == 0) {
      // atomicCAS acts as a load acquire, need TF to enforce ordering
      __threadfence();
      acq1 = true;
    }
  }
  __syncthreads();

  if (!acq1) { return false; } // return if we couldn't acquire the lock
  if (isMasterThread)
  {
    acq2 = false;
    /*
      NOTE: currCount is only accessed by 1 WG at a time and has a lock around
      it, so we can safely access it as a regular data access instead of with
      atomics.
    */
    unsigned int currSemCount = currCount[0];

    if (isWriter) {
      // writer needs the count to be == maxSemCount to enter the critical
      // section (otherwise there are readers in the critical section)
      if (currSemCount == maxSemCount) { acq2 = true; }
    } else {
      // if there is a writer waiting, readers aren't allowed to enter the
      // critical section
      if (writerWaiting[0] == 0) {
        // readers need count > 1 to enter critical section (otherwise semaphore
        // is full)
        if (currSemCount > 1) { acq2 = true; }
      }
    }
  }
  __syncthreads();

  if (!acq2) // release the sem head "lock" since the semaphore was full
  {
    // writers set a flag to note that they are waiting so more readers don't
    // join after the writer started waiting
    if (isWriter) { writerWaiting[0] = 1; /* if already 1, just reset to 1 */ }

    if (isMasterThread) {
      // atomicExch acts as a store release, need TF to enforce ordering
      __threadfence();
      atomicExch(lock, 0);
    }
    __syncthreads();
    return false;
  }
  __syncthreads();

  if (isMasterThread) {
    /*
      NOTE: currCount is only accessed by 1 WG at a time and has a lock around
      it, so we can safely access it as a regular data access instead of with
      atomics.
    */
    if (isWriter) {
      /*
        writer decrements the current count of the semaphore by the max to
        ensure that no one else can enter the critical section while it's
        writing.
      */
      currCount[0] -= maxSemCount;

      // writers also need to unset the "writer is waiting" flag
      writerWaiting[0] = 0;
    } else {
      /*
        readers decrement the current count of the semaphore by 1 so other
        readers can also read the data (but not the writers since they needs
        the entire CS).
      */
      --currCount[0];
    }

    // atomicExch acts as a store release, need TF to enforce ordering
    __threadfence();
    // now that we've updated the semaphore count can release the lock
    atomicExch(lock, 0);
  }
  __syncthreads();

  return true;
}

inline __device__ void hipSemaphoreSpinWait(const hipSemaphore_t sem,
                                            const bool isWriter,
                                            const unsigned int maxSemCount,
                                            unsigned int * semaphoreBuffers,
                                            const int NUM_CU)
{
  while (!hipSemaphoreSpinTryWait(sem, isWriter, maxSemCount, semaphoreBuffers, NUM_CU))
  {
    __syncthreads();
  }
}

inline __device__ void hipSemaphoreSpinPost(const hipSemaphore_t sem,
                                            const bool isWriter,
                                            const unsigned int maxSemCount,
                                            unsigned int * semaphoreBuffers,
                                            const int NUM_CU)
{
  const bool isMasterThread = (threadIdx.x == 0 && threadIdx.y == 0 &&
                               threadIdx.z == 0);
  /*
    Each sem has NUM_CU * 4 locations in the buffer.  Of these locations, each
    CU uses 4 of them (current count, head, tail, max count).  For the global
    semaphore use semaphoreBuffers[sem * 4 * NUM_CU].
  */
  unsigned int * const currCount = semaphoreBuffers + (sem * 4 * NUM_CU);
  unsigned int * const lock = currCount + 1;
  __shared__ bool acquired;

  if (isMasterThread) { acquired = false; }
  __syncthreads();

  while (!acquired)
  {
    if (isMasterThread)
    {
      /*
        NOTE: This CAS will trigger an invalidation since we overload CAS's.
        Since most of the data in the local critical section is written, it
        hopefully won't affect performance too much.
      */
      // try to acquire sem head lock
      if (atomicCAS(lock, 0, 1) == 0) {
        // atomicCAS acts as a load acquire, need TF to enforce ordering
        __threadfence();
        acquired = true;
      }
      else                            { acquired = false; }
    }
    __syncthreads();
  }
  __syncthreads();

  if (isMasterThread) {
    /*
      NOTE: currCount is only accessed by 1 WG at a time and has a lock around
      it, so we can safely access it as a regular data access instead of with
      atomics.
    */
    if (isWriter) {
      // writers add the max value to the semaphore to allow the readers to
      // start accessing the critical section.
      currCount[0] += maxSemCount;
    } else {
      // readers add 1 to the semaphore
      ++currCount[0];
    }

    // atomicExch acts as a store release, need TF to enforce ordering
    __threadfence();
    // now that we've updated the semaphore count can release the lock
    atomicExch(lock, 0);
  }
  __syncthreads();
}

// same wait algorithm but with local scope and per-CU synchronization
inline __device__ bool hipSemaphoreSpinTryWaitLocal(const hipSemaphore_t sem,
                                                    const unsigned int cuID,
                                                    const bool isWriter,
                                                    const unsigned int maxSemCount,
                                                    unsigned int * semaphoreBuffers,
                                                    const int NUM_CU)
{
  const bool isMasterThread = (threadIdx.x == 0 && threadIdx.y == 0 &&
                               threadIdx.z == 0);
  // Each sem has NUM_CU * 4 locations in the buffer.  Of these locations, each
  // CU gets 4 of them (current count, head, tail, max count).  So CU 0 starts
  // at semaphoreBuffers[sem * 4 * NUM_CU].
  unsigned int * const currCount = semaphoreBuffers +
                                       ((sem * 4 * NUM_CU) + (cuID * 4));
  unsigned int * const lock = currCount + 1;
  /*
    Reuse the tail for the "writers are waiting" flag since tail is unused.

    For now just use to indicate that at least 1 writer is waiting instead of
    a count to make sure that readers aren't totally starved out until all the
    writers are done.
  */
  unsigned int * const writerWaiting = currCount + 2;
  __shared__ bool acq1, acq2;

  __syncthreads();
  if (isMasterThread)
  {
    acq1 = false;
    // try to acquire the sem head "lock"
    if (atomicCAS(lock, 0, 1) == 0) {
      // atomicCAS acts as a load acquire, need TF to enforce ordering locally
      __threadfence_block();
      acq1 = true;
    }
  }
  __syncthreads();

  if (!acq1) { return false; } // return if we couldn't acquire the lock
  if (isMasterThread)
  {
    acq2 = false;
    /*
      NOTE: currCount is only accessed by 1 WG at a time and has a lock around
      it, so we can safely access it as a regular data access instead of with
      atomics.
    */
    unsigned int currSemCount = currCount[0];

    if (isWriter) {
      // writer needs the count to be == maxSemCount to enter the critical
      // section (otherwise there are readers in the critical section)
      if (currSemCount == maxSemCount) { acq2 = true; }
    } else {
      // if there is a writer waiting, readers aren't allowed to enter the
      // critical section
      if (writerWaiting[0] == 0) {
        // readers need count > 1 to enter critical section (otherwise semaphore
        // is full)
        if (currSemCount > 1) { acq2 = true; }
      }
    }
  }
  __syncthreads();

  if (!acq2) // release the sem head "lock" since the semaphore was full
  {
    // writers set a flag to note that they are waiting so more readers don't
    // join after the writer started waiting
    if (isWriter) { writerWaiting[0] = 1; /* if already 1, just reset to 1 */ }

    if (isMasterThread) {
      // atomicExch acts as a store release, need TF to enforce ordering locally
      __threadfence_block();
      atomicExch(lock, 0);
    }
    __syncthreads();
    return false;
  }
  __syncthreads();

  if (isMasterThread) {
    /*
      NOTE: currCount is only accessed by 1 WG at a time and has a lock around
      it, so we can safely access it as a regular data access instead of with
      atomics.
    */
    if (isWriter) {
      /*
        writer decrements the current count of the semaphore by the max to
        ensure that no one else can enter the critical section while it's
        writing.
      */
      currCount[0] -= maxSemCount;

      // writers also need to unset the "writer is waiting" flag
      writerWaiting[0] = 0;
    } else {
      /*
        readers decrement the current count of the semaphore by 1 so other
        readers can also read the data (but not the writers since they needs
        the entire CS).
      */
      --currCount[0];
    }

    // atomicExch acts as a store release, need TF to enforce ordering locally
    __threadfence_block();
    // now that we've updated the semaphore count can release the lock
    atomicExch(lock, 0);
  }
  __syncthreads();

  return true;
}

inline __device__ void hipSemaphoreSpinWaitLocal(const hipSemaphore_t sem,
                                                 const unsigned int cuID,
                                                 const bool isWriter,
                                                 const unsigned int maxSemCount,
                                                 unsigned int * semaphoreBuffers,
                                                 const int NUM_CU)
{
  while (!hipSemaphoreSpinTryWaitLocal(sem, cuID, isWriter, maxSemCount, semaphoreBuffers, NUM_CU))
  {
    __syncthreads();
  }
}

inline __device__ void hipSemaphoreSpinPostLocal(const hipSemaphore_t sem,
                                                 const unsigned int cuID,
                                                 const bool isWriter,
                                                 const unsigned int maxSemCount,
                                                 unsigned int * semaphoreBuffers,
                                                 const int NUM_CU)
{
  bool isMasterThread = (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0);
  // Each sem has NUM_CU * 4 locations in the buffer.  Of these locations, each
  // CU gets 4 of them.  So CU 0 starts at semaphoreBuffers[sem * 4 * NUM_CU].
  unsigned int * const currCount = semaphoreBuffers +
                                       ((sem * 4 * NUM_CU) + (cuID * 4));
  unsigned int * const lock = currCount + 1;
  __shared__ bool acquired;

  if (isMasterThread) { acquired = false; }
  __syncthreads();

  while (!acquired)
  {
    if (isMasterThread)
    {
      /*
        NOTE: This CAS will trigger an invalidation since we overload CAS's.
        Since most of the data in the local critical section is written, it
        hopefully won't affect performance too much.
      */
      // try to acquire sem head lock
      if (atomicCAS(lock, 0, 1) == 0) {
        // atomicCAS acts as a load acquire, need TF to enforce ordering locally
        __threadfence_block();
        acquired = true;
      }
      else                            { acquired = false; }
    }
    __syncthreads();
  }
  __syncthreads();

  if (isMasterThread) {
    /*
      NOTE: currCount is only accessed by 1 WG at a time and has a lock around
      it, so we can safely access it as a regular data access instead of with
      atomics.
    */
    if (isWriter) {
      // writers add the max value to the semaphore to allow the readers to
      // start accessing the critical section.
      currCount[0] += maxSemCount;
    } else {
      // readers add 1 to the semaphore
      ++currCount[0];
    }

    // atomicExch acts as a store release, need TF to enforce ordering locally
    __threadfence_block();
    // now that we've updated the semaphore count can release the lock
    atomicExch(lock, 0);
  }
  __syncthreads();
}

inline __device__ bool hipSemaphoreSpinTryWaitPriority(const hipSemaphore_t sem,
                                                       const bool isWriter,
                                                       const unsigned int maxSemCount,
                                                       unsigned int * semaphoreBuffers,
                                                       const int NUM_CU)
{
  const bool isMasterThread = (threadIdx.x == 0 && threadIdx.y == 0 &&
                               threadIdx.z == 0);
  /*
    Each sem has NUM_CU * 5 locations in the buffer.  Of these locations, each
    SM uses 5 of them (current count, head, tail, max count, priority).  For the global
    semaphore all SMs use semaphoreBuffers[sem * 5 * NUM_CU].
  */
  unsigned int * const currCount = semaphoreBuffers + (sem * 5 * NUM_CU);
  unsigned int * const lock = currCount + 1;
  /*
    Reuse the tail for the "writers are waiting" flag since tail is unused.

    For now just use to indicate that at least 1 writer is waiting instead of
    a count to make sure that readers aren't totally starved out until all the
    writers are done.
  */
  unsigned int * const writerWaiting = currCount + 2;
  unsigned int * const priority = currCount + 4;
  __shared__ int backoff;
  __shared__ bool acq1, acq2;

  if (isMasterThread) {
    backoff = 1;
  }
  __syncthreads();
  if (isMasterThread)
  {
    acq1 = false;
    while (atomicCAS(priority, 0, 0) != 0) {
      // Spinning until all blocks wanting to exit the semaphore have exited
      sleepFunc(backoff);
      // Increase backoff to avoid repeatedly hammering priority flag
      backoff = (((backoff << 1) + 1) & (MAX_BACKOFF-1));
    }
    // try to acquire the sem head "lock"
    if (atomicCAS(lock, 0, 1) == 0) {
      // atomicCAS acts as a load acquire, need TF to enforce ordering
      __threadfence();
      acq1 = true;
    }
  }
  __syncthreads();

  if (!acq1) { return false; } // return if we couldn't acquire the lock
  if (isMasterThread)
  {
    acq2 = false;
    /*
      NOTE: currCount is only accessed by 1 WG at a time and has a lock around
      it, so we can safely access it as a regular data access instead of with
      atomics.
    */
    unsigned int currSemCount = currCount[0];

    if (isWriter) {
      // writer needs the count to be == maxSemCount to enter the critical
      // section (otherwise there are readers in the critical section)
      if (currSemCount == maxSemCount) { acq2 = true; }
    } else {
      // if there is a writer waiting, readers aren't allowed to enter the
      // critical section
      if (writerWaiting[0] == 0) {
        // readers need count > 1 to enter critical section (otherwise semaphore
        // is full)
        if (currSemCount > 1) { acq2 = true; }
      }
    }
  }
  __syncthreads();

  if (!acq2) // release the sem head "lock" since the semaphore was full
  {
    // writers set a flag to note that they are waiting so more readers don't
    // join after the writer started waiting
    if (isWriter) { writerWaiting[0] = 1; /* if already 1, just reset to 1 */ }

    if (isMasterThread) {
      // atomicExch acts as a store release, need TF to enforce ordering
      __threadfence();
      atomicExch(lock, 0);
    }
    __syncthreads();
    return false;
  }
  __syncthreads();

  if (isMasterThread) {
    /*
      NOTE: currCount is only accessed by 1 WG at a time and has a lock around
      it, so we can safely access it as a regular data access instead of with
      atomics.
    */
    if (isWriter) {
      /*
        writer decrements the current count of the semaphore by the max to
        ensure that no one else can enter the critical section while it's
        writing.
      */
      currCount[0] -= maxSemCount;

      // writers also need to unset the "writer is waiting" flag
      writerWaiting[0] = 0;
    } else {
      /*
        readers decrement the current count of the semaphore by 1 so other
        readers can also read the data (but not the writers since they needs
        the entire CS).
      */
      --currCount[0];
    }

    // atomicExch acts as a store release, need TF to enforce ordering
    __threadfence();
    // now that we've updated the semaphore count can release the lock
    atomicExch(lock, 0);
  }
  __syncthreads();

  return true;
}

inline __device__ void hipSemaphoreSpinWaitPriority(const hipSemaphore_t sem,
                                                    const bool isWriter,
                                                    const unsigned int maxSemCount,
                                                    unsigned int * semaphoreBuffers,
                                                    const int NUM_CU)
{
  while (!hipSemaphoreSpinTryWaitPriority(sem, isWriter, maxSemCount, semaphoreBuffers, NUM_CU))
  {
    __syncthreads();
  }
}

inline __device__ void hipSemaphoreSpinPostPriority(const hipSemaphore_t sem,
                                                    const bool isWriter,
                                                    const unsigned int maxSemCount,
                                                    unsigned int * semaphoreBuffers,
                                                    const int NUM_CU)
{
  const bool isMasterThread = (threadIdx.x == 0 && threadIdx.y == 0 &&
                               threadIdx.z == 0);
  /*
    Each sem has NUM_CU * 5 locations in the buffer.  Of these locations, each
    SM uses 5 of them (current count, head, tail, max count, priority).  For the global
    semaphore use semaphoreBuffers[sem * 5 * NUM_CU].
  */
  unsigned int * const currCount = semaphoreBuffers + (sem * 5 * NUM_CU);
  unsigned int * const lock = currCount + 1;
  unsigned int * const priority = currCount + 4;
  __shared__ bool acquired;

  if (isMasterThread) 
  { 
    acquired = false;
    /*
    Incrementing priority count whenever a work group wants to exit 
    the Semaphore. A priority count of > 0 will stop blocks trying to enter 
    the semaphore from making an attempt to acquire the lock, reducing contention
    */
    atomicAdd(priority, 1);
  }
  __syncthreads();

  while (!acquired)
  {
    if (isMasterThread)
    {
      /*
        NOTE: This CAS will trigger an invalidation since we overload CAS's.
        Since most of the data in the local critical section is written, it
        hopefully won't affect performance too much.
      */
      // try to acquire sem head lock
      if (atomicCAS(lock, 0, 1) == 0) {
        // atomicCAS acts as a load acquire, need TF to enforce ordering
        __threadfence();
        acquired = true;
      }
      else                            { acquired = false; }
    }
    __syncthreads();
  }
  __syncthreads();

  if (isMasterThread) {
    /*
      NOTE: currCount is only accessed by 1 WG at a time and has a lock around
      it, so we can safely access it as a regular data access instead of with
      atomics.
    */
    if (isWriter) {
      // writers add the max value to the semaphore to allow the readers to
      // start accessing the critical section.
      currCount[0] += maxSemCount;
    } else {
      // readers add 1 to the semaphore
      ++currCount[0];
    }

    // atomicExch acts as a store release, need TF to enforce ordering
    __threadfence();
    // now that we've updated the semaphore count can release the lock
    atomicExch(lock, 0);
    // Decrement priority as work group which wanted to exit has relenquished the lock
    atomicSub(priority, 1);
  }
  __syncthreads();
}

// same wait algorithm but with local scope and per-CU synchronization
inline __device__ bool hipSemaphoreSpinTryWaitLocalPriority (const hipSemaphore_t sem,
                                                              const unsigned int cuID,
                                                              const bool isWriter,
                                                              const unsigned int maxSemCount,
                                                              unsigned int * semaphoreBuffers,
                                                              const int NUM_CU)
{
  const bool isMasterThread = (threadIdx.x == 0 && threadIdx.y == 0 &&
                               threadIdx.z == 0);
  // Each sem has NUM_CU * 5 locations in the buffer.  Of these locations, each
  // SM gets 5 of them (current count, head, tail, max count, priority).  So SM 0 starts
  // at semaphoreBuffers[sem * 5 * NUM_CU].
  unsigned int * const currCount = semaphoreBuffers +
                                       ((sem * 5 * NUM_CU) + (cuID * 5));
  unsigned int * const lock = currCount + 1;
  /*
    Reuse the tail for the "writers are waiting" flag since tail is unused.

    For now just use to indicate that at least 1 writer is waiting instead of
    a count to make sure that readers aren't totally starved out until all the
    writers are done.
  */
  unsigned int * const writerWaiting = currCount + 2;
  unsigned int * const priority = currCount + 4;
  __shared__ int backoff;
  __shared__ bool acq1, acq2;

  if (isMasterThread) {
    backoff = 1;
  }
  __syncthreads();
  if (isMasterThread)
  {
    acq1 = false;
    while(atomicCAS(priority, 0, 0) !=0){
      // Spinning until all blocks wanting to exit the semaphore have exited
      sleepFunc(backoff);
      // Increase backoff to avoid repeatedly hammering priority flag
      backoff = (((backoff << 1) + 1) & (MAX_BACKOFF-1));
    }
    // try to acquire the sem head "lock"
    if (atomicCAS(lock, 0, 1) == 0) {
      // atomicCAS acts as a load acquire, need TF to enforce ordering locally
      __threadfence_block();
      acq1 = true;
    }
  }
  __syncthreads();

  if (!acq1) { return false; } // return if we couldn't acquire the lock
  if (isMasterThread)
  {
    acq2 = false;
    /*
      NOTE: currCount is only accessed by 1 WG at a time and has a lock around
      it, so we can safely access it as a regular data access instead of with
      atomics.
    */
    unsigned int currSemCount = currCount[0];

    if (isWriter) {
      // writer needs the count to be == maxSemCount to enter the critical
      // section (otherwise there are readers in the critical section)
      if (currSemCount == maxSemCount) { acq2 = true; }
    } else {
      // if there is a writer waiting, readers aren't allowed to enter the
      // critical section
      if (writerWaiting[0] == 0) {
        // readers need count > 1 to enter critical section (otherwise semaphore
        // is full)
        if (currSemCount > 1) { acq2 = true; }
      }
    }
  }
  __syncthreads();

  if (!acq2) // release the sem head "lock" since the semaphore was full
  {
    // writers set a flag to note that they are waiting so more readers don't
    // join after the writer started waiting
    if (isWriter) { writerWaiting[0] = 1; /* if already 1, just reset to 1 */ }

    if (isMasterThread) {
      // atomicExch acts as a store release, need TF to enforce ordering locally
      __threadfence_block();
      atomicExch(lock, 0);
    }
    __syncthreads();
    return false;
  }
  __syncthreads();

  if (isMasterThread) {
    /*
      NOTE: currCount is only accessed by 1 WG at a time and has a lock around
      it, so we can safely access it as a regular data access instead of with
      atomics.
    */
    if (isWriter) {
      /*
        writer decrements the current count of the semaphore by the max to
        ensure that no one else can enter the critical section while it's
        writing.
      */
      currCount[0] -= maxSemCount;

      // writers also need to unset the "writer is waiting" flag
      writerWaiting[0] = 0;
    } else {
      /*
        readers decrement the current count of the semaphore by 1 so other
        readers can also read the data (but not the writers since they needs
        the entire CS).
      */
      --currCount[0];
    }

    // atomicExch acts as a store release, need TF to enforce ordering locally
    __threadfence_block();
    // now that we've updated the semaphore count can release the lock
    atomicExch(lock, 0);
  }
  __syncthreads();

  return true;
}

inline __device__ void hipSemaphoreSpinWaitLocalPriority(const hipSemaphore_t sem,
                                                         const unsigned int cuID,
                                                         const bool isWriter,
                                                         const unsigned int maxSemCount,
                                                         unsigned int * semaphoreBuffers,
                                                         const int NUM_CU)
{
  while (!hipSemaphoreSpinTryWaitLocalPriority(sem, cuID, isWriter, maxSemCount, semaphoreBuffers, NUM_CU))
  {
    __syncthreads();
  }
}

inline __device__ void hipSemaphoreSpinPostLocalPriority(const hipSemaphore_t sem,
                                                         const unsigned int cuID,
                                                         const bool isWriter,
                                                         const unsigned int maxSemCount,
                                                         unsigned int * semaphoreBuffers,
                                                         const int NUM_CU)
{
  bool isMasterThread = (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0);
  // Each sem has NUM_CU * 5 locations in the buffer.  Of these locations, each
  // CU gets 5 of them.  So CU 0 starts at semaphoreBuffers[sem * 5 * NUM_CU].
  unsigned int * const currCount = semaphoreBuffers +
                                       ((sem * 5 * NUM_CU) + (cuID * 5));
  unsigned int * const lock = currCount + 1;
  unsigned int * const priority = currCount + 4;
  __shared__ bool acquired;

  if (isMasterThread) 
  { 
    acquired = false;
    /*
      Incrementing priority count whenever a work group wants to exit 
      the Semaphore. A priority count of > 0 will stop blocks trying to enter 
      the semaphore from making an attempt to acquire the lock, reducing contention
    */
    atomicAdd(priority, 1); 
  }
  __syncthreads();

  while (!acquired)
  {
    if (isMasterThread)
    {
      /*
        NOTE: This CAS will trigger an invalidation since we overload CAS's.
        Since most of the data in the local critical section is written, it
        hopefully won't affect performance too much.
      */
      // try to acquire sem head lock
      if (atomicCAS(lock, 0, 1) == 0) {
        // atomicCAS acts as a load acquire, need TF to enforce ordering locally
        __threadfence_block();
        acquired = true;
      }
      else                            { acquired = false; }
    }
    __syncthreads();
  }
  __syncthreads();

  if (isMasterThread) {
    /*
      NOTE: currCount is only accessed by 1 WG at a time and has a lock around
      it, so we can safely access it as a regular data access instead of with
      atomics.
    */
    if (isWriter) {
      // writers add the max value to the semaphore to allow the readers to
      // start accessing the critical section.
      currCount[0] += maxSemCount;
    } else {
      // readers add 1 to the semaphore
      ++currCount[0];
    }

    // atomicExch acts as a store release, need TF to enforce ordering locally
    __threadfence_block();
    // now that we've updated the semaphore count can release the lock
    atomicExch(lock, 0);
    // Decrement priority as work group which wanted to exit has relenquished the lock
    atomicSub(priority, 1);
  }
  __syncthreads();
}

#endif // #ifndef __HIPLOCKSSEMAPHORESPIN_H__
