// Principle Component Background Subtraction 

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __device__
#define CUDA_KERNEL_MEMBER __global__
#else
#define CUDA_CALLABLE_MEMBER
#define CUDA_KERNEL_MEMBER
#endif

// CUDA includes
#include <cuda_runtime.h>
#include <vector_types.h>
#include <cusolverDn.h>

#include "cublas_v2.h"

#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <string>

typedef unsigned char uint8_t; 
typedef unsigned short uint16_t;

extern "C"
void Static_PCBS(int width, 
		  		 int height,
		  		 int cols,					// e.g. number of frames
		  		 float* modelFrames,
		  		 float* targetFrame);