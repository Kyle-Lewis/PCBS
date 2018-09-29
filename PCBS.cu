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
#include "cublas_v2.h"

#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <string>

typedef unsigned char uint8_t; 
typedef unsigned short uint16_t;

// Error check wrapper
inline 
cudaError_t gpuErrchk(cudaError_t result){
	if (result != cudaSuccess){
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result==cudaSuccess);
	}
	return result;
}

// internal, potentially python side pre-processing
// However, given how often we will be running this, 
// it probably belongs here in a C loop. 
// TODO 
// void removeTemporalAvg()

extern "C"
void Static_PCBS(int width, 
		  		 int height,
		  		 int depth,
		  		 float* modelFrames,
		  		 float* targetFrame,
		  		 )
{
	// Create 
	cusolverDnHandle_t cusolverH;
	gpuErrchk(cusolverDnCreate(&solverHandle));


	gpuErrchk(cusolverDnDestroy(&solverHandle));

}

/* The dynamic version is meant to run with a "moving window". 
 * Simply, every call should add a frame to the model block 
 * for consideration in the eigen decomposition. If the model is 
extern "C"
void Dynamic_PCBS(int width, 
		  		 int height,
		  		 int depth,
		  		 float* modelFrames,
		  		 float* targetFrame,
		  		 )
{
	// Create 
	cusolverDnHandle_t cusolverH;
	gpuErrchk(cusolverDnCreate(&solverHandle));


	gpuErrchk(cusolverDnDestroy(&solverHandle));

}