#include <cstdio>
#include <assert.h>
#include "splitCounter_kernel.cu"

int main(int argc, char ** argv) {
  // local variables
  unsigned int * h_counters = NULL;
  unsigned int * h_outArr = NULL;
  const int numRuns = 1;
  int numTBs = 0, tbSize = 0;

  if (argc != 3) {
    fprintf(stderr, "./splitCounter <numTBs> <tbSize>\n");
    fprintf(stderr, "where:\n");
    fprintf(stderr, "\t<numTBs>: number of thread blocks to launch\n");
    fprintf(stderr, "\t<tbSize>: number of threads in a thread block\n");
    exit(-1);
  }

  // parse input args
  numTBs = atoi(argv[1]);
  tbSize = atoi(argv[2]);
  assert(tbSize <= 256); // scratchpad size limited to 256 for 8 TBs to execute

  unsigned int numThrs = (numTBs * tbSize);

  fprintf(stdout, "Initializing data...\n");
  fprintf(stdout, "...allocating memory.\n");
  // every other thread in each TB gets its own counter
  cudaMallocManaged(&h_counters, (numThrs/2)*sizeof(unsigned int));
  // each thread gets its own location in the output array too
  cudaMallocManaged(&h_outArr, numThrs*sizeof(unsigned int));

  // initialize arrays
  fprintf(stdout, "...initializing memory.\n");
  for (int i = 0; i < (numThrs/2); ++i) {
    h_counters[i] = 0;
  }
  for (int i = 0; i < numThrs; ++i) {
    h_outArr[i] = 0;
  }

  fprintf(stdout,
          "Launching kernel - %d runs with %d TBs and %d threads/TB\n",
          numRuns, numTBs, tbSize);
  for (int iter = 0; iter < numRuns; ++iter) {
    splitCounter_kernel<<<numTBs, tbSize>>>(h_counters,
                                            h_outArr);
    cudaDeviceSynchronize();
  }

  bool passFail = true;
  // each repeat of the kernel adds INC_VAL to the counter
  for (int i = 0; i < (numThrs/2); ++i) {
    if (h_counters[i] != numRuns*INC_VAL) {
      fprintf(stderr, "ERROR: h_counters[%d] != %d, = %u\n",
              i, numRuns, h_counters[i]);
      passFail = false;
    }
  }

  // for now the half-warps doing the reads always go first, so they return 0
  // if there are multiple runs, then we should have some partial sum from
  // the previous kernel (assuming these half-warps still execute first,
  // (numRuns-1)*INC_VAL*numCounters where numCounters == numThrs/2)
  int expectedVal = ((numRuns-1)*INC_VAL)*(numThrs/2);
  for (int i = 0; i < numThrs; ++i) {
    if (h_outArr[i] != expectedVal) {
      fprintf(stderr, "\tThread %d: %u, != %d\n", i, h_outArr[i], expectedVal);
      passFail = false;
    }
  }

  if (passFail) { fprintf(stdout, "PASSED\n"); }
  else { fprintf(stdout, "FAILED\n"); }

#ifdef DEBUG
  // print the final values of the counters and the output array
  fprintf(stdout, "Counter Values:\n");
  for (int i = 0; i < (numThrs/2); ++i) {
    fprintf(stdout, "\t[%d] = %u\n", i, h_counters[i]);
  }

  fprintf(stdout, "Per-Thread Output Values\n");
  for (int i = 0; i < numThrs; ++i) {
    fprintf(stdout, "\tThread %d: %u\n", i, h_outArr[i]);
  }
#endif // #ifdef DEBUG

  cudaFreeHost(h_counters);
  cudaFreeHost(h_outArr);
  return 0;
}
