// Principle Component Background Subtraction 

#include "PCBS.h"
#include <iostream>

// Error check wrapper
inline 
cudaError_t gpuErrchk(cudaError_t result){
	if (result != cudaSuccess){
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result==cudaSuccess);
	}
	return result;
}

// inline void cuSolveErrchk(cusolverStatus_t err, const char *file, const int line)
// {
// 	if (CUSOLVER_STATUS_SUCCESS != err) {
// 		fprintf(stderr, "CUSOLVE error in file '%s', line %d, error: %s \nterminating!\n", __FILE__, __LINE__, \
// 			_cusolverGetErrorEnum(err)); \
// 			assert(0); \
// 	}
// }

// internal, potentially python side pre-processing
// However, given how often we will be running this, 
// it probably belongs here in a C loop. 
// TODO 
// void removeTemporalAvg()


/*
 *
 *
 * All references to LDA in the typical documentation are the same as rows here.
 */
extern "C"
void SVD(int width, 
  		 int height,
  		 int cols,					// e.g. number of frames
  		 float* modelFrames,
  		 float* targetFrame)
{
	int rows = width*height;
	float* hS;			// Host array for 
	float* dA; 			// Device ptr to Matrix A in the typical A = U x Sigma x VH ... SVD equation. 
	float* dU;
	float* dV;
	float* dS;
	int work;
	float* devWork;
	int *devInfo;

	// The flattened frames preceeding the target frame 
	gpuErrchk(cudaMalloc(&dA, rows*cols*sizeof(float)));
	gpuErrchk(cudaMemcpy(dA, modelFrames, rows*cols*sizeof(float), cudaMemcpyHostToDevice));

	// The host space for storing the singular values, which we want to check against 
	 = (float*)malloc(std::min(rows, cols)*sizeof(float));

	// The device space for storing the Singular Vectors (columns of U and V)
	// as well as the Singular Values (S is just the diagonals of the Sigma matrix)

	gpuErrchk(cudaMalloc(&dU, rows*rows*sizeof(float))); // Unitary numRows X numRows matrix
	gpuErrchk(cudaMalloc(&dV, rows*cols*sizeof(float))); // 
	gpuErrchk(cudaMalloc(&dS, std::min(rows, cols)*sizeof(float)));

	// Create solver instance
	cusolverDnHandle_t solverHandle;
	cusolverDnCreate(&solverHandle);

	// Get/allocate the amount of working space required for the algorithm through the API 
	cusolverDnDgesvd_bufferSize(solverHandle, rows, cols, &work);
	gpuErrchk(cudaMalloc(&devWork, work * sizeof(float)));
	gpuErrchk(cudaDeviceSynchronize());

	// To check success. 
	gpuErrchk(cudaMalloc(&devInfo, sizeof(int)));

	// Call it 
	auto error = cusolverDnSgesvd(solverHandle,
				     'A', 'A',
				     rows,
				     cols,
				     dA,
				     rows,
				     dS,
				     dU,
				     rows,
				     dV,
				     cols,
				     devWork,
				     work,
				     NULL,
				     devInfo);

	gpuErrchk(cudaDeviceSynchronize());
	std::cout << "happened" << std::endl;

	int hostInfo = 0;
	gpuErrchk(cudaMemcpy(&hostInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
	std::cout << "happened" << hostInfo << std::endl;

	if (hostInfo != 0) 
	{
		std::cout << "SVD device execution failed" << std::endl;
	}
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(hS, dS, std::min(rows, cols) * sizeof(float), cudaMemcpyDeviceToHost));

	std::cout << "Singular values\n";
	for(int i = 0; i < min(rows, cols); i++)
	{
		std::cout << "dS["<<i<<"] = " << hS[i] << std::endl;
	}

	// Free stuff
	// TODO manage this python side, we need them later.
    if (dA      ) cudaFree(dA);
    if (dS      ) cudaFree(dS);
    if (dU      ) cudaFree(dU);
    if (dV      ) cudaFree(dV);
    if (devInfo ) cudaFree(devInfo);
    if (devWork ) cudaFree(devWork);
	cusolverDnDestroy(solverHandle);
}

/* The dynamic version is meant to run with a "moving window". 
 * Simply, every call should add a frame to the model block 
 * for consideration in the eigen decomposition. If the model is */
// extern "C"
// void Dynamic_PCBS(int width, 
// 		  		 int height,
// 		  		 int depth,
// 		  		 float* modelFrames,
// 		  		 float* targetFrame,
// 		  		 )
// {
// 	// Create 
// 	cusolverDnHandle_t cusolverH;
// 	gpuErrchk(cusolverDnCreate(&solverHandle));

// 	gpuErrchk(cusolverDnDestroy(&solverHandle));
// }