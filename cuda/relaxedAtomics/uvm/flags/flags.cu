#include <cstdio>
#include <assert.h>
#include "cuda_error.h"

#define WARP_SIZE 32
#define HALF_WARP_SIZE (WARP_SIZE >> 1)
#define UINT32_MAX ((unsigned int)-1)
/* the input array in the .so file is 64KB * 10, can't have more accesses */
#define MAX_NUM_ACCS ((64 * 1024) * 10)

#include <cudaLocks.cu>
#include <cudaLocksBarrierFast.cu>
#include <flags_kernel.cu>

// program globals
int NUM_SM = 0; // number of SMs our GPU has
int MAX_BLOCKS = 0;

int main(int argc, char ** argv) {
  // local variables
  unsigned int * h_outArr = NULL;
  unsigned int * dirty = NULL, * stop = NULL;
  unsigned int * perSMBarriers = NULL;
  float * randArr = NULL;
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

  // determine number of SMs and max TB/SM
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  NUM_SM = deviceProp.multiProcessorCount;
  /* 7 TBs per SM is current limit given scratchpad allocations */
  MAX_BLOCKS = 7 * NUM_SM;

#ifdef DEBUG
  const int maxTBPerSM = deviceProp.maxThreadsPerBlock/NUM_THREADS_PER_BLOCK;
  fprintf(stdout, "# SM: %d, Max Thrs/TB: %d, Max TB/SM: %d, Max # TB: %d\n",
          NUM_SM, deviceProp.maxThreadsPerBlock, maxTBPerSM, MAX_BLOCKS);
#endif

  fprintf(stdout, "Initializing data...\n");
  fprintf(stdout, "...allocating memory.\n");
  // each thread gets its own location in the output array
  cudaMallocManaged(&h_outArr, dataArrElts*sizeof(unsigned int));
  // each thread gets its own location in the random array
  cudaMallocManaged(&randArr, (numAccs*sizeof(float)));
  /*
    The barriers need a per-SM barrier that is not part of the global synch
    structure.  In terms of size, it needs to be sized to hold the maximum
    number of TBs/SM and each SM needs 2 arrays (for the lock-free version,
    the atomic version needs 1 array).
  */
  cudaMallocManaged(&perSMBarriers, sizeof(unsigned int) * (NUM_SM * MAX_BLOCKS * 2));
  // each flag gets a separate "array" of size 16 (to force them to
  // be on different cache lines)
  cudaMallocManaged(&dirty, 16 * sizeof(unsigned int));
  cudaMallocManaged(&stop, 16 * sizeof(unsigned int));
  // initialize barrier array -- *4 to provide some extra space
  cudaLocksInit(MAX_BLOCKS*4, NUM_SM, pageAlign);
  // initialize rand
  srand(2018);

  // initialize arrays
  fprintf(stdout, "...initializing memory.\n");
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
                                     numIters,
                                     MAX_BLOCKS,
                                     NUM_SM);
    cudaDeviceSynchronize();
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

  cudaFreeHost(h_outArr);
  cudaFreeHost(randArr);
  cudaFreeHost(perSMBarriers);
  cudaFreeHost(dirty);
  cudaFreeHost(stop);
  cudaLocksDestroy();
  return 0;
}
