#include <cstdio>
#include <assert.h>
#include "seqlocks_kernel.cu"

int main(int argc, char ** argv) {
  // local variables
  unsigned int * h_seqlock = NULL, * h_seqlock_temp = NULL;
  int * h_dataArr0 = NULL, * h_dataArr0_temp = NULL;
  int * h_dataArr1 = NULL, * h_dataArr1_temp = NULL;
  int * h_outArr = NULL, * h_outArr_temp = NULL;
  bool pageAlign = false, useTFs = false;
  const int numRuns = 1;
  int numTBs = 0, tbSize = 0, groupSize_seqlock = 0;

  if (argc != 6) {
    fprintf(stderr, "./seqlocks <numTBs> <tbSize> <groupSize_seqlock> <pageAlign> <useTFs>\n");
    fprintf(stderr, "where:\n");
    fprintf(stderr, "\t<numTBs>: number of thread blocks to launch\n");
    fprintf(stderr, "\t<tbSize>: number of threads in a thread block\n");
    fprintf(stderr, "\t<groupSize_seqlock>: how many TBs share a seqlock\n");
    fprintf(stderr, "\t<pageAlign>: if 1 the arrays will be page aligned, else arrays will be unaligned.\n");
    fprintf(stderr, "\t<useTFs>: if 1, use weaker version with more fully relaxed atomics and TFs to enforce ordering\n");
    exit(-1);
  }

  // parse input args
  numTBs = atoi(argv[1]);
  tbSize = atoi(argv[2]);
  groupSize_seqlock = atoi(argv[3]);
  pageAlign = (atoi(argv[4]) == 1);
  useTFs = (atoi(argv[5]) == 1);

  int numThrs = (numTBs * tbSize);
  // want to group TBs together into a few seqlocks to reduce contention
  int numSeqlocks = (numTBs / groupSize_seqlock);

  /*
  // get regions
  // h_seqlocks holds the seqlock, which is written with relaxed atomics and SC
  // atomics (if using CAS_weak) or only SC atomics (if not using CAS_weak).
  // Region varies with this.
  region_t seqReg = (useTFs ? RELAX_ATOM_REGION : SPECIAL_REGION);
  // data1 and data2, which are written with relaxed atomics and SC
  // atomics.  Need to use TFs to get around this problem
  region_t dataReg = RELAX_ATOM_REGION;
  // h_outArr is written in the kernel with data accesses, and uses special region
  region_t outReg = SPECIAL_REGION;
  */

  fprintf(stdout, "Initializing data...\n");
  fprintf(stdout, "...allocating CPU memory.\n");
  h_seqlock_temp         = (unsigned int *)malloc((numSeqlocks * sizeof(unsigned int)) + 0x1000/*, seqReg*/);
  // all threads within a warp read-write a unique location, but all warps
  // are reading-writing the same 32 locations
  h_dataArr0_temp         = (int *)malloc((WARP_SIZE * sizeof(int)) + 0x1000/*, dataReg*/);
  h_dataArr1_temp         = (int *)malloc((WARP_SIZE * sizeof(int)) + 0x1000/*, dataReg*/);
  // each thread gets its own location in the output array
  h_outArr_temp          = (int *)malloc((numThrs * sizeof(int)) + 0x1000/*, outReg*/);
  if (pageAlign) {
    h_seqlock = (unsigned int *)(((((unsigned long long)h_seqlock_temp) >> 12) << 12) + 0x1000);
    h_dataArr0 = (int *)(((((unsigned long long)h_dataArr0_temp) >> 12) << 12) + 0x1000);
    h_dataArr1 = (int *)(((((unsigned long long)h_dataArr1_temp) >> 12) << 12) + 0x1000);
    h_outArr = (int *)(((((unsigned long long)h_outArr_temp) >> 12) << 12) + 0x1000);
  } else {
    h_seqlock = h_seqlock_temp;
    h_dataArr0 = h_dataArr0_temp;
    h_dataArr1 = h_dataArr1_temp;
    h_outArr = h_outArr_temp;
  }

  // initialize arrays
  fprintf(stdout, "...initializing CPU memory.\n");
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

  /*
  // wrote to all 3 regions on CPU, so they need an epilogue
  __denovo_epilogue(3, seqReg, outReg, dataReg);
  */

  fprintf(stdout,
          "Launching kernel - %d runs with %d TBs and %d threads/TB\n",
          numRuns, numTBs, tbSize);
  for (int iter = 0; iter < numRuns; ++iter) {
    if (useTFs) {
      seqlocks_kernel_tfs<<<numTBs, tbSize>>>(h_seqlock,
                                              h_dataArr0,
                                              h_dataArr1,
                                              h_outArr,
                                              groupSize_seqlock/*,
                                              seqReg,
                                              dataReg,
                                              outReg*/);
    } else {
      seqlocks_kernel<<<numTBs, tbSize>>>(h_seqlock,
                                          h_dataArr0,
                                          h_dataArr1,
                                          h_outArr,
                                          groupSize_seqlock/*,
                                          seqReg,
                                          dataReg,
                                          outReg*/);
    }

    /*
    // kernel writes all 3 arrays, so need to do an epilogue on them
    __denovo_epilogue(3, seqReg, dataReg, outReg);
    */
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

  free(h_seqlock);
  free(h_dataArr0);
  free(h_dataArr1);
  free(h_outArr);
  return 0;
}
