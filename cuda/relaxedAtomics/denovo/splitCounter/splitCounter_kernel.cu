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
  // for manual loop unrolling
  unsigned int localSum0 = 0; // rest of local sums are done internally in the inlined assembly
  unsigned int currLoc0 = 0, currLoc1 = 0, currLoc2 = 0, currLoc3 = 0,
               currLoc4 = 0, currLoc5 = 0, currLoc6 = 0, currLoc7 = 0,
               currLoc8 = 0, currLoc9 = 0, currLoc10 = 0, currLoc11 = 0,
               currLoc12 = 0, currLoc13 = 0, currLoc14 = 0, currLoc15 = 0;
  unsigned int * addr0, * addr1, * addr2, * addr3, * addr4, * addr5, * addr6,
               * addr7, * addr8, * addr9, * addr10, * addr11, * addr12,
               * addr13, * addr14, * addr15;

  localSum[threadIdx.x] = 0; // each thread initializes its local sum to 0
  __syncthreads();

  /*
    Each thread takes a chunk of the counters and adds them into a scratchpad
    array -- since CUDA doesn't have an atomic load, do an add of 0 to get the
    current counter value (these can be relaxed).

    We manually unroll the loop to increase the number of atomics that can be
    overlapped.
  */
  // ** NOTE: This code does not handle stragglers, so if numAccsPerThr is a
  // not multiple of 16, 8, 4, or 2, then it does the non-unrolled version
  if ((numAccsPerThr % 16) == 0) {
    // need to initialize each of the currLoci variables to their appropriate
    // location in the array
    currLoc0 = currLoc;
    currLoc1 = ((currLoc0 + numCounters_perTB) % numCounters);
    currLoc2 = ((currLoc1 + numCounters_perTB) % numCounters);
    currLoc3 = ((currLoc2 + numCounters_perTB) % numCounters);
    currLoc4 = ((currLoc3 + numCounters_perTB) % numCounters);
    currLoc5 = ((currLoc4 + numCounters_perTB) % numCounters);
    currLoc6 = ((currLoc5 + numCounters_perTB) % numCounters);
    currLoc7 = ((currLoc6 + numCounters_perTB) % numCounters);
    currLoc8 = ((currLoc7 + numCounters_perTB) % numCounters);
    currLoc9 = ((currLoc8 + numCounters_perTB) % numCounters);
    currLoc10 = ((currLoc9 + numCounters_perTB) % numCounters);
    currLoc11 = ((currLoc10 + numCounters_perTB) % numCounters);
    currLoc12 = ((currLoc11 + numCounters_perTB) % numCounters);
    currLoc13 = ((currLoc12 + numCounters_perTB) % numCounters);
    currLoc14 = ((currLoc13 + numCounters_perTB) % numCounters);
    currLoc15 = ((currLoc14 + numCounters_perTB) % numCounters);

    int i = 0;
    for (i = 0; i < numAccsPerThr; i += 16) {
      // need to set the addresses before the inlined assembly or else a bunch
      // of address calculation instructions will be inserted in between the
      // atomics
      addr0 = &(counters[currLoc0]);
      addr1 = &(counters[currLoc1]);
      addr2 = &(counters[currLoc2]);
      addr3 = &(counters[currLoc3]);
      addr4 = &(counters[currLoc4]);
      addr5 = &(counters[currLoc5]);
      addr6 = &(counters[currLoc6]);
      addr7 = &(counters[currLoc7]);
      addr8 = &(counters[currLoc8]);
      addr9 = &(counters[currLoc9]);
      addr10 = &(counters[currLoc10]);
      addr11 = &(counters[currLoc11]);
      addr12 = &(counters[currLoc12]);
      addr13 = &(counters[currLoc13]);
      addr14 = &(counters[currLoc14]);
      addr15 = &(counters[currLoc15]);

      // all unrolled atomics -- can be overlapped and reordered
      // ** NOTE: Across all of the inlined assembly blocks we can't reuse the
      // same temp reg names
      asm volatile(// Temp Registers
                   ".reg .u32 q1;\n\t"  // temp reg q1 (localSum0)
                   ".reg .u32 q2;\n\t"  // temp reg q2 (localSum1)
                   ".reg .u32 q3;\n\t"  // temp reg q3 (localSum2)
                   ".reg .u32 q4;\n\t"  // temp reg q4 (localSum3)
                   ".reg .u32 q5;\n\t"  // temp reg q5 (localSum4)
                   ".reg .u32 q6;\n\t"  // temp reg q6 (localSum5)
                   ".reg .u32 q7;\n\t"  // temp reg q7 (localSum6)
                   ".reg .u32 q8;\n\t"  // temp reg q8 (localSum7)
                   ".reg .u32 q9;\n\t"  // temp reg q9 (localSum8)
                   ".reg .u32 q10;\n\t" // temp reg q10 (localSum9)
                   ".reg .u32 q11;\n\t" // temp reg q11 (localSum10)
                   ".reg .u32 q12;\n\t" // temp reg q12 (localSum11)
                   ".reg .u32 q13;\n\t" // temp reg q13 (localSum12)
                   ".reg .u32 q14;\n\t" // temp reg q14 (localSum13)
                   ".reg .u32 q15;\n\t" // temp reg q15 (localSum14)
                   ".reg .u32 q16;\n\t" // temp reg q16 (localSum15)
                   ".reg .u32 q17;\n\t" // temp reg q17 (localSum0 + localSum1)
                   ".reg .u32 q18;\n\t" // temp reg q18 (localSum2 + localSum3)
                   ".reg .u32 q19;\n\t" // temp reg q19 (localSum4 + localSum5)
                   ".reg .u32 q20;\n\t" // temp reg q20 (localSum6 + localSum7)
                   ".reg .u32 q21;\n\t" // temp reg q21 (localSum8 + localSum9)
                   ".reg .u32 q22;\n\t" // temp reg q22 (localSum10 + localSum11)
                   ".reg .u32 q23;\n\t" // temp reg q23 (localSum12 + localSum13)
                   ".reg .u32 q24;\n\t" // temp reg q24 (localSum14 + localSum15)
                   ".reg .u32 q25;\n\t" // temp reg q25 (sum(localSum[0:3]))
                   ".reg .u32 q26;\n\t" // temp reg q26 (sum(localSum[4:7]))
                   ".reg .u32 q27;\n\t" // temp reg q27 (sum(localSum[8:11]))
                   ".reg .u32 q28;\n\t" // temp reg q28 (sum(localSum[12:15]))
                   ".reg .u32 q29;\n\t" // temp reg q29 (sum(localSum[0:7]))
                   ".reg .u32 q30;\n\t" // temp reg q30 (sum(localSum[8:15]))
                   // PTX Instructions
                   // ** NOTE: even though the atomic adds return the old value,
                   // since we're adding 0 old == new
                   "atom.add.u32 q1, [%1], 0;\n\t"   // atomicAdd for addr0
                   "atom.add.u32 q2, [%2], 0;\n\t"   // atomicAdd for addr1
                   "atom.add.u32 q3, [%3], 0;\n\t"   // atomicAdd for addr2
                   "atom.add.u32 q4, [%4], 0;\n\t"   // atomicAdd for addr3
                   "atom.add.u32 q5, [%5], 0;\n\t"   // atomicAdd for addr4
                   "atom.add.u32 q6, [%6], 0;\n\t"   // atomicAdd for addr5
                   "atom.add.u32 q7, [%7], 0;\n\t"   // atomicAdd for addr6
                   "atom.add.u32 q8, [%8], 0;\n\t"   // atomicAdd for addr7
                   "atom.add.u32 q9, [%9], 0;\n\t"   // atomicAdd for addr8
                   "atom.add.u32 q10, [%10], 0;\n\t" // atomicAdd for addr9
                   "atom.add.u32 q11, [%11], 0;\n\t" // atomicAdd for addr10
                   "atom.add.u32 q12, [%12], 0;\n\t" // atomicAdd for addr11
                   "atom.add.u32 q13, [%13], 0;\n\t" // atomicAdd for addr12
                   "atom.add.u32 q14, [%14], 0;\n\t" // atomicAdd for addr13
                   "atom.add.u32 q15, [%15], 0;\n\t" // atomicAdd for addr14
                   "atom.add.u32 q16, [%16], 0;\n\t" // atomicAdd for addr15
                   "add.u32 q17, q1, q2;\n\t"        // localSum0 + localSum1
                   "add.u32 q18, q3, q4;\n\t"        // localSum2 + localSum3
                   "add.u32 q19, q5, q6;\n\t"        // localSum4 + localSum5
                   "add.u32 q20, q7, q8;\n\t"        // localSum6 + localSum7
                   "add.u32 q21, q9, q10;\n\t"       // localSum8 + localSum9
                   "add.u32 q22, q11, q12;\n\t"      // localSum10 + localSum11
                   "add.u32 q23, q13, q14;\n\t"      // localSum12 + localSum13
                   "add.u32 q24, q15, q16;\n\t"      // localSum14 + localSum15
                   "add.u32 q25, q17, q18;\n\t"      // sum(localSum[0:3])
                   "add.u32 q26, q19, q20;\n\t"      // sum(localSum[4:7])
                   "add.u32 q27, q21, q22;\n\t"      // sum(localSum[8:11])
                   "add.u32 q28, q23, q24;\n\t"      // sum(localSum[12:15])
                   "add.u32 q29, q25, q26;\n\t"      // sum(localSum[0:7])
                   "add.u32 q30, q27, q28;\n\t"      // sum(localSum[8:15])
                   "add.u32 %0, q29, q30;"           // sum(localSum[0:15])
                   // outputs (put in localSum0, then store in localSum)
                   : "=r"(localSum0)
                   // inputs
                   : "l"(addr0), "l"(addr1), "l"(addr2), "l"(addr3),
                     "l"(addr4), "l"(addr5), "l"(addr6), "l"(addr7),
                     "l"(addr8), "l"(addr9), "l"(addr10), "l"(addr11),
                     "l"(addr12), "l"(addr13), "l"(addr14), "l"(addr15)
                   );

      // update the local scratchpad sum with the partial sum
      localSum[threadIdx.x] += localSum0;
      __syncthreads();

      // the next accesses are numCounters_perTB away to coalesce the accesses
      // in a half-warp in a given iteration
      currLoc0 = ((currLoc0 + numCounters_perTB) % numCounters);
      currLoc1 = ((currLoc1 + numCounters_perTB) % numCounters);
      currLoc2 = ((currLoc2 + numCounters_perTB) % numCounters);
      currLoc3 = ((currLoc3 + numCounters_perTB) % numCounters);
      currLoc4 = ((currLoc4 + numCounters_perTB) % numCounters);
      currLoc5 = ((currLoc5 + numCounters_perTB) % numCounters);
      currLoc6 = ((currLoc6 + numCounters_perTB) % numCounters);
      currLoc7 = ((currLoc7 + numCounters_perTB) % numCounters);
      currLoc8 = ((currLoc8 + numCounters_perTB) % numCounters);
      currLoc9 = ((currLoc9 + numCounters_perTB) % numCounters);
      currLoc10 = ((currLoc10 + numCounters_perTB) % numCounters);
      currLoc11 = ((currLoc11 + numCounters_perTB) % numCounters);
      currLoc12 = ((currLoc12 + numCounters_perTB) % numCounters);
      currLoc13 = ((currLoc13 + numCounters_perTB) % numCounters);
      currLoc14 = ((currLoc14 + numCounters_perTB) % numCounters);
      currLoc15 = ((currLoc15 + numCounters_perTB) % numCounters);
    }
  } else if ((numAccsPerThr % 8) == 0) {
    // need to initialize each of the currLoci variables to their appropriate
    // location in the array
    currLoc0 = currLoc;
    currLoc1 = ((currLoc0 + numCounters_perTB) % numCounters);
    currLoc2 = ((currLoc1 + numCounters_perTB) % numCounters);
    currLoc3 = ((currLoc2 + numCounters_perTB) % numCounters);
    currLoc4 = ((currLoc3 + numCounters_perTB) % numCounters);
    currLoc5 = ((currLoc4 + numCounters_perTB) % numCounters);
    currLoc6 = ((currLoc5 + numCounters_perTB) % numCounters);
    currLoc7 = ((currLoc6 + numCounters_perTB) % numCounters);

    for (int i = 0; i < numAccsPerThr; i += 8) {
      // need to set the addresses before the inlined assembly or else a bunch
      // of address calculation instructions will be inserted in between the
      // atomics
      addr0 = &(counters[currLoc0]);
      addr1 = &(counters[currLoc1]);
      addr2 = &(counters[currLoc2]);
      addr3 = &(counters[currLoc3]);
      addr4 = &(counters[currLoc4]);
      addr5 = &(counters[currLoc5]);
      addr6 = &(counters[currLoc6]);
      addr7 = &(counters[currLoc7]);

      // all unrolled atomics -- can be overlapped and reordered
      // ** NOTE: Across all of the inlined assembly blocks we can't reuse the
      // same temp reg names
      asm volatile(// Temp Registers
                   ".reg .u32 n1;\n\t"  // temp reg n1 (localSum0)
                   ".reg .u32 n2;\n\t"  // temp reg n2 (localSum1)
                   ".reg .u32 n3;\n\t"  // temp reg n3 (localSum2)
                   ".reg .u32 n4;\n\t"  // temp reg n4 (localSum3)
                   ".reg .u32 n5;\n\t"  // temp reg n5 (localSum4)
                   ".reg .u32 n6;\n\t"  // temp reg n6 (localSum5)
                   ".reg .u32 n7;\n\t"  // temp reg n7 (localSum6)
                   ".reg .u32 n8;\n\t"  // temp reg n8 (localSum7)
                   ".reg .u32 n9;\n\t"  // temp reg n9 (localSum0 + localSum1)
                   ".reg .u32 n10;\n\t" // temp reg n10 (localSum2 + localSum3)
                   ".reg .u32 n11;\n\t" // temp reg n11 (localSum4 + localSum5)
                   ".reg .u32 n12;\n\t" // temp reg n12 (localSum6 + localSum7)
                   ".reg .u32 n13;\n\t" // temp reg n13 (sum(localSum[0:3]))
                   ".reg .u32 n14;\n\t" // temp reg n14 (sum(localSum[4:7]))
                   // PTX Instructions
                   // ** NOTE: even though the atomic adds return the old value,
                   // since we're adding 0 old == new
                   "atom.add.u32 n1, [%1], 0;\n\t" // atomicAdd for addr0
                   "atom.add.u32 n2, [%2], 0;\n\t" // atomicAdd for addr1
                   "atom.add.u32 n3, [%3], 0;\n\t" // atomicAdd for addr2
                   "atom.add.u32 n4, [%4], 0;\n\t" // atomicAdd for addr3
                   "atom.add.u32 n5, [%5], 0;\n\t" // atomicAdd for addr4
                   "atom.add.u32 n6, [%6], 0;\n\t" // atomicAdd for addr5
                   "atom.add.u32 n7, [%7], 0;\n\t" // atomicAdd for addr6
                   "atom.add.u32 n8, [%8], 0;\n\t" // atomicAdd for addr7
                   "add.u32 n9, n1, n2;\n\t"       // localSum0 + localSum1
                   "add.u32 n10, n3, n4;\n\t"      // localSum2 + localSum3
                   "add.u32 n11, n5, n6;\n\t"      // localSum4 + localSum5
                   "add.u32 n12, n7, n8;\n\t"      // localSum6 + localSum7
                   "add.u32 n13, n9, n10;\n\t"     // sum(localSum[0:3])
                   "add.u32 n14, n11, n12;\n\t"    // sum(localSum[4:7])
                   "add.u32 %0, n13, n14;"         // sum(localSum[0:7])
                   // outputs (put in localSum0, then store in localSum)
                   : "=r"(localSum0)
                   // inputs
                   : "l"(addr0), "l"(addr1), "l"(addr2), "l"(addr3),
                     "l"(addr4), "l"(addr5), "l"(addr6), "l"(addr7)
                   );

      // update the local scratchpad sum with these values
      localSum[threadIdx.x] += localSum0;
      __syncthreads();

      // the next accesses are numCounters_perTB away to coalesce the accesses
      // in a half-warp in a given iteration
      currLoc0 = ((currLoc0 + numCounters_perTB) % numCounters);
      currLoc1 = ((currLoc1 + numCounters_perTB) % numCounters);
      currLoc2 = ((currLoc2 + numCounters_perTB) % numCounters);
      currLoc3 = ((currLoc3 + numCounters_perTB) % numCounters);
      currLoc4 = ((currLoc4 + numCounters_perTB) % numCounters);
      currLoc5 = ((currLoc5 + numCounters_perTB) % numCounters);
      currLoc6 = ((currLoc6 + numCounters_perTB) % numCounters);
      currLoc7 = ((currLoc7 + numCounters_perTB) % numCounters);
    }
  } else if ((numAccsPerThr % 4) == 0) {
    // need to initialize each of the currLoci variables to their appropriate
    // location in the array
    currLoc0 = currLoc;
    currLoc1 = ((currLoc0 + numCounters_perTB) % numCounters);
    currLoc2 = ((currLoc1 + numCounters_perTB) % numCounters);
    currLoc3 = ((currLoc2 + numCounters_perTB) % numCounters);

    for (int i = 0; i < numAccsPerThr; i += 4) {
      // need to set the addresses before the inlined assembly or else a bunch
      // of address calculation instructions will be inserted in between the
      // atomics
      addr0 = &(counters[currLoc0]);
      addr1 = &(counters[currLoc1]);
      addr2 = &(counters[currLoc2]);
      addr3 = &(counters[currLoc3]);

      // all unrolled atomics -- can be overlapped and reordered
      // ** NOTE: Across all of the inlined assembly blocks we can't reuse the
      // same temp reg names
      asm volatile(// Temp Registers
                   ".reg .u32 m1;\n\t" // temp reg m1 (localSum0)
                   ".reg .u32 m2;\n\t" // temp reg m2 (localSum1)
                   ".reg .u32 m3;\n\t" // temp reg m3 (localSum2)
                   ".reg .u32 m4;\n\t" // temp reg m4 (localSum3)
                   ".reg .u32 m5;\n\t" // temp reg m5 (localSum0 + localSum1)
                   ".reg .u32 m6;\n\t" // temp reg m6 (localSum2 + localSum3)
                   // PTX Instructions
                   // ** NOTE: even though the atomic adds return the old value,
                   // since we're adding 0 old == new
                   "atom.add.u32 m1, [%1], 0;\n\t" // atomicAdd for addr0
                   "atom.add.u32 m2, [%2], 0;\n\t" // atomicAdd for addr1
                   "atom.add.u32 m3, [%3], 0;\n\t" // atomicAdd for addr2
                   "atom.add.u32 m4, [%4], 0;\n\t" // atomicAdd for addr3
                   "add.u32 m5, m1, m2;\n\t"       // localSum0 + localSum0
                   "add.u32 m6, m3, m4;\n\t"       // localSum2 + localSum3
                   "add.u32 %0, m5, m6;"           // sum(localSum[0:3]
                   // outputs (put in localSum0, then store in localSum)
                   : "=r"(localSum0)
                   // inputs
                   : "l"(addr0), "l"(addr1), "l"(addr2), "l"(addr3)
                   );

      // update the local scratchpad sum with the partial sum
      localSum[threadIdx.x] += localSum0;
      __syncthreads();

      // the next accesses are numCounters_perTB away to coalesce the accesses
      // in a half-warp in a given iteration
      currLoc0 = ((currLoc0 + numCounters_perTB) % numCounters);
      currLoc1 = ((currLoc1 + numCounters_perTB) % numCounters);
      currLoc2 = ((currLoc2 + numCounters_perTB) % numCounters);
      currLoc3 = ((currLoc3 + numCounters_perTB) % numCounters);
    }
  } else if ((numAccsPerThr % 2) == 0) {
    // need to initialize each of the currLoci variables to their appropriate
    // location in the array
    currLoc0 = currLoc;
    currLoc1 = ((currLoc0 + numCounters_perTB) % numCounters);

    for (int i = 0; i < numAccsPerThr; i += 2) {
      // need to set the addresses before the inlined assembly or else a bunch
      // of address calculation instructions will be inserted in between the
      // atomics
      addr0 = &(counters[currLoc0]);
      addr1 = &(counters[currLoc1]);

      // all unrolled atomics -- can be overlapped and reordered
      // ** NOTE: Across all of the inlined assembly blocks we can't reuse the
      // same temp reg names
      asm volatile(// Temp Registers
                   ".reg .u32 t1;\n\t"    // temp reg t1 (localSum0)
                   ".reg .u32 t2;\n\t"    // temp reg t2 (localSum1)
                   // PTX Instructions
                   // ** NOTE: even though the atomic adds return the old value,
                   // since we're adding 0 old == new
                   "atom.add.u32 t1, [%1], 0;\n\t" // atomicAdd for addr0
                   "atom.add.u32 t2, [%2], 0;\n\t" // atomicAdd for addr1
                   "add.u32 %0, t1, t2;"           // localSum0 + localSum1
                   // outputs (put in localSum0, then store in localSum)
                   : "=r"(localSum0)
                   // inputs
                   : "l"(addr0), "l"(addr1)
                   );

      // update the local scratchpad sum with the partial sum
      localSum[threadIdx.x] += localSum0;
      __syncthreads();

      // the next accesses are numCounters_perTB away to coalesce the accesses
      // in a half-warp in a given iteration
      currLoc0 = ((currLoc0 + numCounters_perTB) % numCounters);
      currLoc1 = ((currLoc1 + numCounters_perTB) % numCounters);
    }
  } else {
    /*
    ** NOTE: Don't need inlined assembly for this case because only 1 atomic,
    so nothing to overlap with relaxed atomics
    */
    for (int i = 0; i < numAccsPerThr; ++i) {
      localSum[threadIdx.x] += atomicAdd(&(counters[currLoc]), 0);
      __syncthreads();

      // the next access is numCounters_perTB away to coalesce the accesses in a
      // half-warp in a given iteration
      currLoc = ((currLoc + numCounters_perTB) % numCounters);
    }
  }

  // to get the final sum we need to sum the local partial sums -- do reduction
  // in scratchpad too (from CUDA SDK reduction example)
  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    if ((threadIdx.x % (2*s)) == 0) { // modulo arithmetic is slow!
      localSum[threadIdx.x] += localSum[threadIdx.x + s];
    }
    __syncthreads();
  }

  return localSum[0];
}

__global__ void splitCounter_kernel(unsigned int * counters,
                                    unsigned int * outArr/*,
                                    region_t countReg,
                                    region_t outReg*/) {
  // local variables
  // every other thread in each TB has its own counter (so readers and writers
  // access the same counters) -- assumes all TBs are equally sized and powers
  // of 2
  const unsigned int numCounters_perTB = (blockDim.x >> 1);
  // get my warp ID within my TB
  const unsigned int warpID = (threadIdx.x / WARP_SIZE);
  // unique thread ID for each thread
  const unsigned int threadID = ((blockIdx.x * blockDim.x) + threadIdx.x);

  /*
  if (threadIdx.x == 0) {
    __denovo_setAcquireRegion(outReg);
    __denovo_addAcquireRegion(countReg); // should only be written with atomics
  }
  __syncthreads();
  */

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
  //__syncthreads();

  /*
  if (threadIdx.x == 0) {
    __denovo_gpuEpilogue(countReg); // written with atomics in add_split_counter
    __denovo_gpuEpilogue(outReg);   // written in main
  }
  */
}
