#include <cstdio>
#include <assert.h>
#include "splitCounter_kernel.cu"

int main(int argc, char ** argv) {
  // local variables
  unsigned int * h_counters = NULL, * h_counters_temp = NULL;
  unsigned int * h_outArr = NULL, * h_outArr_temp = NULL;
  bool pageAlign = false;
  const int numRuns = 1;
  int numTBs = 0, tbSize = 0;

  if (argc != 4) {
    fprintf(stderr, "./splitCounter <numTBs> <tbSize> <pageAlign>\n");
    fprintf(stderr, "where:\n");
    fprintf(stderr, "\t<numTBs>: number of thread blocks to launch\n");
    fprintf(stderr, "\t<tbSize>: number of threads in a thread block\n");
    fprintf(stderr, "\t<pageAlign>: if 1 the arrays will be page aligned, else arrays will be unaligned.\n");
    exit(-1);
  }

  // parse input args
  numTBs = atoi(argv[1]);
  tbSize = atoi(argv[2]);
  assert(tbSize <= 256); // scratchpad size limited to 256 for 8 TBs to execute
  pageAlign = (atoi(argv[3]) == 1);

  unsigned int numThrs = (numTBs * tbSize);

  /*
  // get regions
  // h_counters holds the counters, which are written with relaxed atomics
  region_t countReg = RELAX_ATOM_REGION;
  // h_outArr is written in the kernel with data accesses, use special region
  region_t outReg = SPECIAL_REGION;
  */

  fprintf(stdout, "Initializing data...\n");
  fprintf(stdout, "...allocating CPU memory.\n");
  // every other thread in each TB gets its own counter
  h_counters_temp         = (unsigned int *)malloc(((numThrs/2)*sizeof(unsigned int)) + 0x1000/*, countReg*/);
  // each thread gets its own location in the output array too
  h_outArr_temp           = (unsigned int *)malloc((numThrs*sizeof(unsigned int)) + 0x1000/*, outReg*/);
  if (pageAlign) {
    h_counters = (unsigned int *)(((((unsigned long long)h_counters_temp) >> 12) << 12) + 0x1000);
    h_outArr = (unsigned int *)(((((unsigned long long)h_outArr_temp) >> 12) << 12) + 0x1000);
  } else {
    h_counters = h_counters_temp;
    h_outArr = h_outArr_temp;
  }

  // initialize arrays
  fprintf(stdout, "...initializing CPU memory.\n");
  for (int i = 0; i < (numThrs/2); ++i) {
    h_counters[i] = 0;
  }
  for (int i = 0; i < numThrs; ++i) {
    h_outArr[i] = 0;
  }

  /*
  // wrote to both regions on CPU, so they need an epilogue
  __denovo_epilogue(2, countReg, outReg);
  */

  // now that the initialization stuff is done, reset the counters and start
  // the simulation!
  fprintf(stdout,
          "Launching kernel - %d runs with %d TBs and %d threads/TB\n",
          numRuns, numTBs, tbSize);
  for (int iter = 0; iter < numRuns; ++iter) {
    splitCounter_kernel<<<numTBs, tbSize>>>(h_counters,
                                            h_outArr/*,
                                            countReg,
                                            outReg*/);

    /*
    // kernel writes counter and output arrays, so need to do an epilogue on them
    __denovo_epilogue(2, countReg, outReg);
    */
  }

  bool passFail = true;
  // each repeat of the kernel adds INC_VAL to the counter
  for (int i = 0; i < (numThrs/2); ++i) {
    if (h_counters[i] != numRuns*INC_VAL) {
      fprintf(stderr, "ERROR: h_counters[%] != %d, = %u\n",
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

  free(h_counters);
  free(h_outArr);
  return 0;
}
