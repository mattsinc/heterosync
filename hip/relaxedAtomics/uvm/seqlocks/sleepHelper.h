#ifndef __SLEEPHELPER_H__
#define __SLEEPHELPER_H__

// used for calling s_sleep
extern "C" void __builtin_amdgcn_s_sleep(int);

/*
  Shared sleep function.  Since s_sleep only takes in consstants (between 1 and 128),
  need code to handle long tail.
 */
inline __device__ void sleepFunc(int backoff) {
  int backoffCopy = backoff;
#ifdef GFX9
  // max for gfx9 is 127
  for (int i = 0; i < backoff; i += 127) {
    __builtin_amdgcn_s_sleep(127);
    backoffCopy -= 127;
  }
#else
  // max for gfx8 is 15
  for (int i = 0; i < backoff; i += 15) {
    __builtin_amdgcn_s_sleep(15);
    backoffCopy -= 15;
  }
#endif

  // handle any additional backoff
#ifdef GFX9
  if (backoffCopy > 64) {
    __builtin_amdgcn_s_sleep(64);
    backoffCopy -= 64;
  }
  if (backoffCopy > 32) {
    __builtin_amdgcn_s_sleep(32);
    backoffCopy -= 32;
  }
  if (backoffCopy > 16) {
    __builtin_amdgcn_s_sleep(16);
    backoffCopy -= 16;
  }
#endif
  if (backoffCopy > 8) {
    __builtin_amdgcn_s_sleep(8);
    backoffCopy -= 8;
  }
  if (backoffCopy > 4) {
    __builtin_amdgcn_s_sleep(4);
    backoffCopy -= 4;
  }
  if (backoffCopy > 2) {
    __builtin_amdgcn_s_sleep(2);
    backoffCopy -= 2;
  }
  if (backoffCopy > 1) {
    __builtin_amdgcn_s_sleep(1);
    backoffCopy -= 1;
  }
}

#endif
