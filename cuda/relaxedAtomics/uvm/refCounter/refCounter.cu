#include <cstdio>
#include <assert.h>
#include "refCounter_kernel.cu"

int main(int argc, char ** argv) {
  // local variables
  unsigned int * h_counters0 = NULL;
  unsigned int * h_counters1 = NULL;
  unsigned int * h_del0 = NULL;
  unsigned int * h_del1 = NULL;
  unsigned int numTBs = 0, tbSize = 0, numRepeats = 0, numSharersPerGroup = 0;
  const int numRuns = 1;

  if (argc != 5) {
    fprintf(stderr, "./refCounter <numSharersPerGroup> <numTBs> <tbSize> <numRepeats>\n");
    fprintf(stderr, "where:\n");
    fprintf(stderr, "\t<numSharersPerGroup>: number of thread blocks sharing a counter\n");
    fprintf(stderr, "\t<numTBs>: number of thread blocks to launch\n");
    fprintf(stderr, "\t<tbSize>: number of threads in a thread block\n");
    fprintf(stderr, "\t<numRepeats>: how many times to have the 'main' TB read its data before setting stop\n");
    exit(-1);
  }

  // parse input args
  numSharersPerGroup = atoi(argv[1]);
  numTBs = atoi(argv[2]);
  assert(numSharersPerGroup <= numTBs);
  tbSize = atoi(argv[3]);
  assert(tbSize <= 256); // so scratchpad allocations don't throttle max TBs/SM too much
  numRepeats = atoi(argv[4]);
  const unsigned int numThrs = (tbSize * numTBs);
  /*
    numSharersPerGroup TBs share a counter entry and a del entry.  Within each
    TB, each thread has a separate counter that is shared with the same thread #
    in the sharing TBs.
  */
  unsigned int numCounters = (numThrs / numSharersPerGroup);
  unsigned int numSharingGroups = (numTBs / numSharersPerGroup);
  unsigned int numCounters_perSharingGroup = (numCounters / numSharingGroups);

  fprintf(stdout,
          "# Thr: %d, # TB: %d, # Counters: %d, # Sharers: %d, # Groups: %d, # Counters/Group: %d\n",
          numThrs, numTBs, numCounters, numSharersPerGroup, numSharingGroups, numCounters_perSharingGroup);

  fprintf(stdout, "Initializing data...\n");
  fprintf(stdout, "...allocating memory.\n");
  cudaMallocManaged(&h_counters0, numCounters*sizeof(int));
  cudaMallocManaged(&h_counters1, numCounters*sizeof(int));
  cudaMallocManaged(&h_del0, numThrs*sizeof(int));
  cudaMallocManaged(&h_del1, numThrs*sizeof(int));

  // initialize arrays
  fprintf(stdout, "...initializing memory.\n");
  for (int i = 0; i < numCounters; ++i) {
    h_counters0[i] = 0;
    h_counters1[i] = 0;
  }
  for (int i = 0; i < numThrs; ++i) {
    h_del0[i] = 0;
    h_del1[i] = 0;
  }

  fprintf(stdout, "Launching kernel - %d runs with %d TBs and %d threads/TB\n",
          numRuns, numTBs, tbSize);
  for (int iter = 0; iter < numRuns; ++iter) {
    refCounter_kernel<<<numTBs, tbSize>>>(h_counters0,
                                          h_counters1,
                                          h_del0,
                                          h_del1,
                                          numRepeats,
                                          numSharersPerGroup,
                                          numCounters,
                                          numSharingGroups,
                                          numCounters_perSharingGroup);
    cudaDeviceSynchronize();
  }

  /*
    Instead of printing all the values, do some simple checks to
    decide if the output is reasonable or not (there is no one right
    output because the interleavings of the threads determine what will happen).
    So just make sure that all of the counters are 0 and that at least one
    location in the del arrays is true
  */
  bool passFail = true;
  for (int i = 0; i < numCounters; ++i) {
    if (h_counters0[i] != 0) {
      fprintf(stderr, "ERROR: h_counters0[%d]: %d\n", i, h_counters0[i]);
      passFail = false;
    }
  }

  for (int i = 0; i < numCounters; ++i) {
    if (h_counters1[i] != 0) {
      fprintf(stderr, "ERROR: h_counters1[%d]: %d\n", i, h_counters1[i]);
      passFail = false;
    }
  }

  bool atLeastOneSet0 = false, atLeastOneSet1 = false;
  for (int i = 0; i < numThrs; ++i) {
    if (h_del0[i] != 0) {
      atLeastOneSet0 = true;
      break;
    }
  }

  if (!atLeastOneSet0) {
    fprintf(stderr, "ERROR: none of d_del0 array locations set\n");
    passFail = false;
  }

  for (int i = 0; i < numThrs; ++i) {
    if (h_del1[i] != 0) {
      atLeastOneSet1 = true;
      break;
    }
  }

  if (!atLeastOneSet1) {
    fprintf(stderr, "ERROR: none of d_del1 array locations set\n");
    passFail = false;
  }

  if (!passFail) { fprintf(stderr, "TEST FAILED!\n"); }
  else { fprintf(stderr, "TEST PASSED!\n"); }

#ifdef DEBUG
  fprintf(stdout, "Counter 0 Values\n");
  for (int i = 0; i < numCounters; ++i) {
    fprintf(stdout, "\t[%d]: %d\n", i, h_counters0[i]);
  }

  fprintf(stdout, "Counter 1 Values\n");
  for (int i = 0; i < numCounters; ++i) {
    fprintf(stdout, "\t[%d]: %d\n", i, h_counters1[i]);
  }

  fprintf(stdout, "Deletion 0 Values\n");
  for (int i = 0; i < numThrs; ++i) {
    fprintf(stdout, "\t[%d]: %s\n", i, ((h_del0[i]) ? "true" : "false"));
  }

  fprintf(stdout, "Deletion 1 Values\n");
  for (int i = 0; i < numThrs; ++i) {
    fprintf(stdout, "\t[%d]: %s\n", i, ((h_del1[i]) ? "true" : "false"));
  }
#endif // #ifdef DEBUG

  cudaFreeHost(h_counters0);
  cudaFreeHost(h_del0);
  cudaFreeHost(h_counters1);
  cudaFreeHost(h_del1);
  return 0;
}
