#include <cstdio>
#include <assert.h>
#include "seqlocks_kernel.cu"

int main(int argc, char ** argv) {
  // local variables
  unsigned int * h_seqlock = NULL;
  int * h_dataArr0 = NULL;
  int * h_dataArr1 = NULL;
  int * h_outArr = NULL;
  bool useTFs = false;
  const int numRuns = 1;
  int numTBs = 0, tbSize = 0, groupSize_seqlock = 0;

  if (argc != 5) {
    fprintf(stderr, "./seqlocks <numTBs> <tbSize> <groupSize_seqlock> <pageAlign> <useTFs>\n");
    fprintf(stderr, "where:\n");
    fprintf(stderr, "\t<numTBs>: number of thread blocks to launch\n");
    fprintf(stderr, "\t<tbSize>: number of threads in a thread block\n");
    fprintf(stderr, "\t<groupSize_seqlock>: how many TBs share a seqlock\n");
    fprintf(stderr, "\t<useTFs>: if 1, use weaker version with more fully relaxed atomics and TFs to enforce ordering\n");
    exit(-1);
  }

  // parse input args
  numTBs = atoi(argv[1]);
  tbSize = atoi(argv[2]);
  groupSize_seqlock = atoi(argv[3]);
  useTFs = (atoi(argv[4]) == 1);

  int numThrs = (numTBs * tbSize);
  // want to group TBs together into a few seqlocks to reduce contention
  int numSeqlocks = (numTBs / groupSize_seqlock);

  fprintf(stdout, "Initializing data...\n");
  fprintf(stdout, "...allocating memory.\n");
  cudaMallocManaged(&h_seqlock, numSeqlocks * sizeof(unsigned int));
  // all threads within a warp read-write a unique location, but all warps
  // are reading-writing the same 32 locations
  cudaMallocManaged(&h_dataArr0, WARP_SIZE * sizeof(int));
  cudaMallocManaged(&h_dataArr1, WARP_SIZE * sizeof(int));
  // each thread gets its own location in the output array
  cudaMallocManaged(&h_outArr, numThrs * sizeof(int));

  // initialize arrays
  fprintf(stdout, "...initializing memory.\n");
  for (int i = 0; i < numSeqlocks; ++i) {
    h_seqlock[i] = 0;
  }
  for (int i = 0; i < WARP_SIZE; ++i) {
    h_dataArr0[i] = 0;
    h_dataArr1[i] = 0;
  }
  for (int i = 0; i < numThrs; ++i) {
    h_outArr[i] = -1;
  }

  fprintf(stdout,
          "Launching kernel - %d runs with %d TBs and %d threads/TB\n",
          numRuns, numTBs, tbSize);
  for (int iter = 0; iter < numRuns; ++iter) {
    if (useTFs) {
      seqlocks_kernel_tfs<<<numTBs, tbSize>>>(h_seqlock,
                                              h_dataArr0,
                                              h_dataArr1,
                                              h_outArr,
                                              groupSize_seqlock);
    } else {
      seqlocks_kernel<<<numTBs, tbSize>>>(h_seqlock,
                                          h_dataArr0,
                                          h_dataArr1,
                                          h_outArr,
                                          groupSize_seqlock);
    }
    cudaDeviceSynchronize();
  }

  // print the final values of the arrays
  for (int i = 0; i < numSeqlocks; ++i) {
#ifdef DEBUG
    fprintf(stdout, "Seqlock[%d] Value: %d\n", i, h_seqlock[i]);
#endif // #ifdef DEBUG
    // seqlock should always be a multiple of 2
    if (h_seqlock[i] % 2 != 0) {
      fprintf(stderr, "ERROR: seqlock should be a multiple of 2!\n");
    }
  }

#ifdef DEBUG
  fprintf(stdout, "Data0 Arr Values:\n");
  for (int i = 0; i < WARP_SIZE; ++i) {
    fprintf(stdout, "dataArr0[%d] = %d\n", i, h_dataArr0[i]);
  }

  fprintf(stdout, "Data1 Arr Values:\n");
  for (int i = 0; i < WARP_SIZE; ++i) {
    fprintf(stdout, "dataArr1[%d] = %d\n", i, h_dataArr1[i]);
  }

  fprintf(stdout, "Per-Thread Output Values\n");
  for (int i = 0; i < numThrs; ++i) {
    fprintf(stdout, "\tThread %d: %d\n", i, h_outArr[i]);
  }
#endif // #ifdef DEBUG

  cudaFreeHost(h_seqlock);
  cudaFreeHost(h_dataArr0);
  cudaFreeHost(h_dataArr1);
  cudaFreeHost(h_outArr);
  return 0;
}
