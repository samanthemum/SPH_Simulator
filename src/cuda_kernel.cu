#include <cuda_runtime.h>
#include "cuda_kernel.cuh"

__global__ void vectorAdditionKernel(double* A, double* B, double* C, int arraySize) {
    // Get thread ID.
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if thread is within array bounds.
    if (threadID < arraySize) {
        // Add a and b.
        C[threadID] = A[threadID] + B[threadID];
    }
}



/**
 * Wrapper function for the CUDA kernel function.
 * @param A Array A.
 * @param B Array B.
 * @param C Sum of array elements A and B directly across.
 * @param arraySize Size of arrays A, B, and C.
 */
void kernel(double* A, double* B, double* C, int arraySize) {

    // Initialize device pointers.
    double* d_A, * d_B, * d_C;

    // Allocate device memory.
    cudaMalloc((void**)&d_A, arraySize * sizeof(double));
    cudaMalloc((void**)&d_B, arraySize * sizeof(double));
    cudaMalloc((void**)&d_C, arraySize * sizeof(double));

    // Transfer arrays a and b to device.
    cudaMemcpy(d_A, A, arraySize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, arraySize * sizeof(double), cudaMemcpyHostToDevice);

    // Calculate blocksize and gridsize.
    dim3 blockSize(512, 1, 1);
    dim3 gridSize(512 / arraySize + 1, 1);

    // Launch CUDA kernel.
    vectorAdditionKernel << <gridSize, blockSize >> > (d_A, d_B, d_C, arraySize);

    // Copy result array c back to host memory.
    cudaMemcpy(C, d_C, arraySize * sizeof(double), cudaMemcpyDeviceToHost);
}

__global__ void setDensitiesForParticles(Particle* particleList, int particleCount) {
    // Get thread ID.
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;

    if (threadID < particleCount) {
        // Calculate the density
        Particle x = particleList[threadID];

        float density = x.getMass() * Kernel::polyKernelFunction(x, x, x.getIsMatchPoint());
        for (int j = 0; j < x.numNeighbors; j++) {
            int index = x.neighborIndices[j];
            density += (particleList[index].getMass() * Kernel::polyKernelFunction(x, particleList[index], x.getIsMatchPoint()));
        }
    }
}

void setDensitiesForParticles_CUDA(Particle* particleList, int particleCount) {
    // Initialize device pointers
    // Particle* d_particleList;

    // Calculate blocksize and grid size
    dim3 blockSize(512, 1, 1);
    dim3 gridSize(512 / particleCount + 1, 1);

    // Launch CUDA kernel.
    //vectorAdditionKernel << <gridSize, blockSize >> > (d_A, d_B, d_C, arraySize);

    // Copy result array c back to host memory.
    // cudaMemcpy(particleList, d_particleList, particleCount * sizeof(Particle), cudaMemcpyDeviceToHost);
}