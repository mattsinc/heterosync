#ifndef CUDA_CHECK_ERROR
#define CUDA_CHECK_ERROR

void inline checkError(cudaError_t cudaErr, const char * functWithError)
{
  if ( cudaErr != cudaSuccess )
  {
    fprintf(stderr, "ERROR %s - %s\n", functWithError,
            cudaGetErrorString(cudaErr));
    exit(-1);
  }
}

#endif
