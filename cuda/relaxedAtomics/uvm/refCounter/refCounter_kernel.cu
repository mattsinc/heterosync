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
    /*
      Replace the above atomics with inlined PTX to ensure that they are
      next to each other in the instruction sequence and thus can be
      overlapped.

      NOTE: Across all of the inlined assembly blocks we can't reuse the
      same temp reg names.
    */
    /*
    asm volatile(// Temp Registers
                 // t1 and t2 aren't actually used for anything (they hold
                 //  the results of the atomic adds, but we don't return
                 // them).  Still need them for correct PTX though.
                 ".reg .u32 t1;\n\t"    // temp reg t1
                 ".reg .u32 t2;\n\t"    // temp reg t2
                 // PTX Instructions
                 "atom.add.u32 t1, [%0], 1;\n\t" // atomicAdd for counterAddr0
                 "atom.add.u32 t2, [%1], 1;"     // atomicAdd for counterAddr1
                 // no outputs
                 // inputs
                 :: "l"(counterAddr0), "l"(counterAddr1)
                 );
    */

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
    /*
      Replace the above atomics with inlined PTX to ensure that they are
      next to each other in the instruction sequence and thus can be
      overlapped.

      NOTE: Across all of the inlined assembly blocks we can't reuse the
      same temp reg names.
    */
    /*
    unsigned int * delAddr0 = NULL, * delAddr1 = NULL;
    delAddr0 = &(d_del0[myBaseLoc]);
    delAddr1 = &(d_del1[myBaseLoc]);
    asm volatile(// Temp Registers
                 ".reg .u32 q3;\n\t"    // temp reg q3 (atomAdd(counterAddr0) result)
                 ".reg .u32 q4;\n\t"    // temp reg q4 (atomAdd(counterAddr1) result))
                 ".reg .pred p5;\n\t"   // temp predicate reg p5 (branch0)
                 ".reg .pred p6;\n\t"   // temp predicate reg p6 (branch1)
                 // PTX Instructions
                 "atom.dec.u32 q4, [%1], 10000000000;\n\t" // atomicDec for counterAddr1 (store release semantics, -1 is 10000000000)
                 "atom.dec.u32 q3, [%0], 10000000000;\n\t" // atomicDec for counterAddr0 (store release semantics, -1 is 10000000000)
                 // can't pass out two values, so need to do ifs here
                 // part1 of branch for counterAddr0 result -- set p5 to 1 if
                 // q3 > q2 (if result of atomicDec is > 1) (since q3 and q4
                 // hold the old values, we check for 1 instead of 0)
                 "setp.gt.u32 p5, q3, 1;\n\t"
                 // part2 of branch for counterAddr0 result -- don't do the
                 // scratchpad store if p5 = 1 (if result of atomicDec is > 0)
                 "@p5 bra $CounterAddr1If;\n\t"
                 // if the first atomic sub result is <= 0, then set d_del0[i]
                 // to 1 (true) -- store to global array, since the scratchpad
                 // array writes don't work properly with inlined PTX
                 "st.global.s32 [%2], 1;\n\t"
                 "$CounterAddr1If:\n\t"           // label for start of counterAddr1 if
                 // part1 of branch for counterAddr1 result -- set p6 to 1 if
                 // q4 > 1 (if result of atomicDec is > 1)  (since q3 and q4
                 // hold the old values, we check for 1 instead of 0)
                 "setp.gt.u32 p6, q4, 1;\n\t"
                 // part2 of branch for counterAddr1 result -- don't do the
                 // scratchpad store if p6 = 1 (if result of atomicDec is > 0)
                 "@p6 bra $Done;\n\t"
                 // if the second atomic sub result is <= 0, then set d_del1[1]
                 // to 1 (true) -- store to global array, since the scratchpad
                 // array writes don't work properly with inlined PTX
                 "st.global.s32 [%3], 1;\n\t"
                 "$Done:"
                 // no outputs
                 // inputs
                 :: "l"(counterAddr0), "l"(counterAddr1), "l"(delAddr0), "l"(delAddr1)
                 );
    */
  }
}
