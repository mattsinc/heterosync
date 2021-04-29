#include <cstdio>
#include <assert.h>

#define NUM_SM 15
#define WARP_SIZE 32
#define HALF_WARP_SIZE (WARP_SIZE >> 1)
#define MAX_BLOCKS (NUM_SM * 7) /* 7 TBs per SM is current limit given scratchpad allocations */
#define UINT32_MAX ((unsigned int)-1)
/* the input array in the .so file is 64KB * 10, can't have more accesses */
#define MAX_NUM_ACCS ((64 * 1024) * 10)

#include <cudaLocks.cu>
#include <cudaLocksBarrierFast.cu>
#include <flags_kernel.cu>

int main(int argc, char ** argv) {
  // local variables
  unsigned int * h_outArr = NULL, * h_outArr_temp = NULL;
  unsigned int * dirty = NULL, * stop = NULL;
  unsigned int * perSMBarriers_temp = NULL, * perSMBarriers = NULL;
  float * randArr_temp = NULL, * randArr = NULL;
  bool pageAlign = false;
  unsigned int numAccsPerThr = 0; // each thread accesses N elements
  unsigned int numTBs = 0, tbSize = 0, numRepeats = 0, numIters = 0;
  float cutoffVal = 0.0f;
  const int numRuns = 1;

  if (argc != 8) {
    fprintf(stderr, "./flags <numTBs> <tbSize> <numAccsPerThr> <numRepeats> <cutoffVal> <numIters> <pageAlign>\n");
    fprintf(stderr, "where:\n");
    fprintf(stderr, "\t<numTBs>: number of thread blocks to launch\n");
    fprintf(stderr, "\t<tbSize>: number of threads in a thread block\n");
    fprintf(stderr, "\t<numAccsPerThr>: number of data accesses each thread does in the kernel\n");
    fprintf(stderr, "\t<numRepeats>: how many times to have the 'main' TB read its data before setting stop\n");
    fprintf(stderr, "\t<cutoffVal>: the cutoff value for comparing the random values to (all > will be accepted)\n");
    fprintf(stderr, "\t<numIters>: number of iterations for worker TB's loop\n");
    fprintf(stderr, "\t<pageAlign>: if 1 the arrays will be page aligned, else arrays will be unaligned.\n");
    exit(-1);
  }

  // parse input args
  numTBs = atoi(argv[1]);
  tbSize = atoi(argv[2]);
  assert(tbSize <= 256);
  numAccsPerThr = atoi(argv[3]);
  numRepeats = atoi(argv[4]);
  cutoffVal = atof(argv[5]);
  assert(cutoffVal >= 0.0f && cutoffVal <= 1.0f); // cutoff [0, 1]
  numIters = atoi(argv[6]);
  pageAlign = (atoi(argv[7]) == 1);
  const unsigned int numThrs = (tbSize * numTBs);
  unsigned int dataArrElts = numThrs;
  const unsigned int numAccs = (numIters * numTBs);
  assert(numAccs <= MAX_NUM_ACCS);

  /*
  // get regions
  // all flags are written with relaxed atomics
  region_t flagsReg = RELAX_ATOM_REGION;
  // h_outArr is written in the kernel with data accesses, use special region
  region_t outReg = SPECIAL_REGION; // special - outArr
  // barrier has global scope, so any region is sufficient
  region_t locksReg = SPECIAL_REGION; // special - barrier
  // the tree barrier has hybrid local-global scheme, so if scopes are used we
  // can obtain additional benefits.
  region_t localLocksReg = SCOPE_LOCAL_REGION;
  // random data is read-only
  region_t randReg = READ_ONLY_REGION;
  */

  fprintf(stdout, "Initializing data...\n");
  fprintf(stdout, "...allocating CPU memory.\n");
  // each thread gets its own location in the output array
  h_outArr_temp           = (unsigned int *)malloc((dataArrElts*sizeof(unsigned int)) + 0x1000/*, outReg*/);
  // each thread gets its own location in the random array
  randArr_temp            = (float *)malloc((numAccs*sizeof(float)) + 0x1000/*, randReg*/);
  /*
    The barriers need a per-SM barrier that is not part of the global synch
    structure.  In terms of size, it needs to be sized to hold the maximum
    number of TBs/SM and each SM needs 2 arrays (for the lock-free version,
    the atomic version needs 1 array).
  */
  perSMBarriers_temp = (unsigned int *)malloc((sizeof(unsigned int) * (NUM_SM * MAX_BLOCKS * 2)) + 0x1000/*, localLocksReg*/);
  // each flag gets a separate "array" of size 16 (to force them to
  // be on different cache lines)
  dirty = (unsigned int *)malloc(16 * sizeof(unsigned int)/*, flagsReg*/);
  stop = (unsigned int *)malloc(16 * sizeof(unsigned int)/*, flagsReg*/);
  if (pageAlign) {
    h_outArr = (unsigned int *)(((((unsigned long long)h_outArr_temp) >> 12) << 12) + 0x1000);
    perSMBarriers = (unsigned int *)(((((unsigned long long)perSMBarriers_temp) >> 12) << 12) + 0x1000);
    randArr = (float *)(((((unsigned long long)randArr_temp) >> 12) << 12) + 0x1000);
  } else {
    h_outArr = h_outArr_temp;
    perSMBarriers = perSMBarriers_temp;
    randArr = randArr_temp;
  }
  // initialize barrier array -- *4 to provide some extra space
  cudaLocksInit(MAX_BLOCKS*4, pageAlign/*, locksReg*/);
  // initialize rand
  srand(2018);

  // initialize arrays
  fprintf(stdout, "...initializing CPU memory.\n");
  for (int i = 0; i < dataArrElts; ++i) {
    h_outArr[i] = 0;
  }
  for (int i = 0; i < numAccs; ++i) {
    randArr[i] = ((float)rand() / (float)UINT32_MAX);
  }
  for (int i = 0; i < 16; ++i) {
    dirty[i] = 0;
    stop[i] = 0;
  }
  for (int i = 0; i < (NUM_SM * MAX_BLOCKS * 2); ++i) { perSMBarriers[i] = 0; }

  /*
  // wrote to all 5 regions on CPU, so they need an epilogue
  __denovo_epilogue(5, flagsReg, outReg, locksReg, localLocksReg, randReg);
  */

  // now that the initialization stuff is done, reset the counters and start
  // the simulation!
  fprintf(stdout,
          "Launching kernel - %d runs with %d TBs and %d threads/TB, %u iterations of worker TB loop\n",
          numRuns, numTBs, tbSize, numIters);
  for (int iter = 0; iter < numRuns; ++iter) {
    flags_kernel<<<numTBs, tbSize>>>(cpuLockData->barrierBuffers,
                                     perSMBarriers,
                                     stop,
                                     dirty,
                                     h_outArr,
                                     randArr,
                                     cpuLockData->arrayStride,
                                     numAccsPerThr,
                                     numRepeats,
                                     cutoffVal,
                                     numIters/*,
                                     locksReg,
                                     localLocksReg,
                                     flagsReg,
                                     outReg,
                                     randReg*/);

    /*
    // kernel writes all 4 arrays, so need to do an epilogue on them (include rand
    // to be safe)
    __denovo_epilogue(5, flagsReg, outReg, locksReg, localLocksReg, randReg);
    */
  }

  fprintf(stdout, "isDirty - %d\n", dirty[0]);

  // check the output
  bool passFail = true;
  const unsigned int expectedVal_tb0 = (numRepeats * numAccsPerThr * numAccsPerThr);
  // if isDirty is 0, then output array should be all 0's
  if (dirty[0] == 0) {
    for (int i = 0; i < dataArrElts; ++i) {
      if (h_outArr[i] != 0) {
        fprintf(stderr, "ERROR: outArr[%d] != 0, = %u\n", i, h_outArr[i]);
        passFail = false;
      }
    }
  } else {
    // if isDirty is 1, then the last TBs part should be non-0 and the rest
    // should be 0
    for (int i = 0; i < dataArrElts; ++i) {
      if ((i >= tbSize * (numTBs-1)) && (i < dataArrElts)) {
        if (h_outArr[i] != expectedVal_tb0) {
          fprintf(stderr, "outArr[%d] != %u, = %u\n", i, expectedVal_tb0, h_outArr[i]);
          passFail = false;
        }
      } else {
        if (h_outArr[i] != 0) {
          fprintf(stderr, "ERROR: outArr[%d] != 0, = %u\n", i, h_outArr[i]);
          passFail = false;
        }
      }
    }
  }
  fprintf(stdout, "Test %s!\n", ((passFail) ? "PASSED" : "FAILED"));

#ifdef DEBUG
  fprintf(stdout, "Output Array Values\n");
  for (int i = 0; i < dataArrElts; ++i) {
    fprintf(stdout, "\t[%d]: %u\n", i, h_outArr[i]);
  }
#endif // #ifdef DEBUG

  free(h_outArr);
  free(dirty);
  free(stop);
  cudaLocksDestroy();
  free(perSMBarriers);
  return 0;
}
