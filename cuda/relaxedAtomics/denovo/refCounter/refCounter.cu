#include <cstdio>
#include <assert.h>
#include "refCounter_kernel.cu"

int main(int argc, char ** argv) {
  // local variables
  int * h_counters0 = NULL, * h_counters0_temp = NULL;
  int * h_counters1 = NULL, * h_counters1_temp = NULL;
  int * h_del0 = NULL, * h_del0_temp = NULL;
  int * h_del1 = NULL, * h_del1_temp = NULL;
  bool pageAlign = false;
  unsigned int numTBs = 0, tbSize = 0, numRepeats = 0, numSharersPerGroup = 0;
  const int numRuns = 1;

  if (argc != 6) {
    fprintf(stderr, "./refCounter <numSharersPerGroup> <numTBs> <tbSize> <numRepeats> <pageAlign>\n");
    fprintf(stderr, "where:\n");
    fprintf(stderr, "\t<numSharersPerGroup>: number of thread blocks sharing a counter\n");
    fprintf(stderr, "\t<numTBs>: number of thread blocks to launch\n");
    fprintf(stderr, "\t<tbSize>: number of threads in a thread block\n");
    fprintf(stderr, "\t<numRepeats>: how many times to have the 'main' TB read its data before setting stop\n");
    fprintf(stderr, "\t<pageAlign>: if 1 the arrays will be page aligned, else arrays will be unaligned.\n");
    exit(-1);
  }

  // parse input args
  numSharersPerGroup = atoi(argv[1]);
  numTBs = atoi(argv[2]);
  assert(numSharersPerGroup <= numTBs);
  tbSize = atoi(argv[3]);
  assert(tbSize <= 256); // so scratchpad allocations don't throttle max TBs/SM too much
  numRepeats = atoi(argv[4]);
  pageAlign = (atoi(argv[5]) == 1);
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

  /*
  // get regions
  // both counter arrays are written with relaxed atomics
  region_t counterReg = RELAX_ATOM_REGION;
  // both del arrays are written with regular stores
  region_t delReg = SPECIAL_REGION; // special - h_del0, h_del1
  */

  fprintf(stdout, "Initializing data...\n");
  fprintf(stdout, "...allocating CPU memory.\n");
  h_counters0_temp           = (int *)malloc((numCounters*sizeof(int)) + 0x1000/*, counterReg*/);
  h_counters1_temp           = (int *)malloc((numCounters*sizeof(int)) + 0x1000/*, counterReg*/);
  h_del0_temp                = (int *)malloc((numThrs*sizeof(int)) + 0x1000/*, delReg*/);
  h_del1_temp                = (int *)malloc((numThrs*sizeof(int)) + 0x1000/*, delReg*/);
  if (pageAlign) {
    h_counters0 = (int *)(((((unsigned long long)h_counters0_temp) >> 12) << 12) + 0x1000);
    h_counters1 = (int *)(((((unsigned long long)h_counters1_temp) >> 12) << 12) + 0x1000);
    h_del0 = (int *)(((((unsigned long long)h_del0_temp) >> 12) << 12) + 0x1000);
    h_del1 = (int *)(((((unsigned long long)h_del1_temp) >> 12) << 12) + 0x1000);
  } else {
    h_counters0 = h_counters0_temp;
    h_counters1 = h_counters1_temp;
    h_del0 = h_del0_temp;
    h_del1 = h_del1_temp;
  }

  // initialize arrays
  fprintf(stdout, "...initializing CPU memory.\n");
  for (int i = 0; i < numCounters; ++i) {
    h_counters0[i] = 0;
    h_counters1[i] = 0;
  }
  for (int i = 0; i < numThrs; ++i) {
    h_del0[i] = 0;
    h_del1[i] = 0;
  }

  /*
  // wrote to both regions on CPU, so they need an epilogue
  __denovo_epilogue(2, counterReg, delReg);
  */

  // now that the initialization stuff is done, reset the counters and start
  // the simulation!
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
                                          numCounters_perSharingGroup/*,
                                          counterReg,
                                          delReg*/);

    /*
    // kernel writes all 4 arrays, so both of their regions need to do an
    // epilogue
    __denovo_epilogue(2, counterReg, delReg);
    */
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

  free(h_counters0);
  free(h_del0);
  free(h_counters1);
  free(h_del1);
  return 0;
}
