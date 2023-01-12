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
void test_kernel(double* A, double* B, double* C, int arraySize) {

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

__global__ void setDensitiesForParticles(Particle* particleList, int particleCount, Kernel* kernel) {
    // Get thread ID.
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;

    if (threadID < particleCount) {
        // Calculate the density
        float density = particleList[threadID].getMass() * kernel->polyKernelFunction(particleList[threadID], particleList[threadID], particleList[threadID].getIsMatchPoint());
        float kernelValue = kernel->polyKernelFunction(particleList[threadID], particleList[threadID], particleList[threadID].getIsMatchPoint());
        for (int j = 0; j < particleList[threadID].numNeighbors; j++) {
            int index = particleList[threadID].neighborIndices[j];
            density += (particleList[index].getMass() * kernel->polyKernelFunction(particleList[threadID], particleList[index], particleList[threadID].getIsMatchPoint()));
        }
       
        particleList[threadID].setDensity(density);
    }
}

void setDensitiesForParticles_CUDA(Particle* particleList, int particleCount, Kernel* kernel) {
    // Initialize device pointers
    // Particle* d_particleList;

    // Calculate blocksize and grid size
    int blockSize = 512;
    int gridSize = (particleCount + blockSize - 1) / blockSize;

    // Launch CUDA kernel.
    //vectorAdditionKernel << <gridSize, blockSize >> > (d_A, d_B, d_C, arraySize);
    setDensitiesForParticles << <gridSize, blockSize >> > (particleList, particleCount, kernel);

    // Copy result array c back to host memory.
    // cudaMemcpy(particleList, d_particleList, particleCount * sizeof(Particle), cudaMemcpyDeviceToHost);
}

__global__ void surfaceNormalField(Particle* particleList, int particleCount, Kernel* kernel) {
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadID < particleCount) {
        // for every Particle xj in the neighbor hood of xi
        glm::vec3 surfaceField = glm::vec3(0.0f, 0.0f, 0.0f);
        for (int j = 0; j < particleList[threadID].numNeighbors; j++) {
            int index = particleList[threadID].neighborIndices[j];

            float outside_term = particleList[index].getMass() * 1 / particleList[index].getDensity();
            surfaceField += (outside_term * kernel->polyKernelGradient(particleList[threadID], particleList[index]));


        }
        particleList[threadID].setSurfaceNormal(surfaceField);
    }
    
}

void setSurfaceNormalFieldForParticles_CUDA(Particle* particleList, int particleCount, Kernel* kernel) {

    // Calculate blocksize and grid size
    int blockSize = 512;
    int gridSize = (particleCount + blockSize - 1) / blockSize;

    // Launch CUDA kernel.
    //vectorAdditionKernel << <gridSize, blockSize >> > (d_A, d_B, d_C, arraySize);
    surfaceNormalField << <gridSize, blockSize >> > (particleList, particleCount, kernel);
}

__global__ void colorFieldLaplacian(Particle* particleList, int particleCount, Kernel* kernel) {
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadID < particleCount) {
        float surfaceField = 0;

        // for every Particle xj in the neighbor hood of xi
        for (int j = 0; j < particleList[threadID].numNeighbors; j++) {
            int index = particleList[threadID].neighborIndices[j];
            float outside_term = particleList[index].getMass() * 1 / particleList[index].getDensity();

            surfaceField += (outside_term * kernel->polyKernelLaplacian(particleList[threadID], particleList[index]));
        }

        particleList[threadID].setColorFieldLaplacian(surfaceField);
    }
}

void setColorFieldLaplaciansForParticles_CUDA(Particle* particleList, int particleCount, Kernel* kernel) {

    // Calculate blocksize and grid size
    int blockSize = 512;
    int gridSize = (particleCount + blockSize - 1) / blockSize;

    // Launch CUDA kernel.
    colorFieldLaplacian << <gridSize, blockSize >> > (particleList, particleCount, kernel);

}

__global__ void calculatePressureForParticle(Particle* particleList, int particleCount, float STIFFNESS_PARAM, float DENSITY_0_GUESS, Kernel* kernel) {
    
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadID < particleCount) {
        float pressure = STIFFNESS_PARAM * (particleList[threadID].getDensity() - DENSITY_0_GUESS);
        particleList[threadID].setPressure(pressure);
    }
}

void setPressuresForParticles_CUDA(Particle* particleList, int particleCount, float STIFFNESS_PARAM, float DENSITY_0_GUESS, Kernel* kernel) {
    // Calculate blocksize and grid size
    int blockSize = 512;
    int gridSize = (particleCount + blockSize - 1) / blockSize;

    // Launch CUDA kernel.
    calculatePressureForParticle << <gridSize, blockSize >> > (particleList, particleCount, STIFFNESS_PARAM, DENSITY_0_GUESS, kernel);
}

__device__ glm::vec3 pressureGradient(int threadID, Particle* particleList, Kernel* kernel) {
    glm::vec3 pressureGradient = glm::vec3(0.0f, 0.0f, 0.0f);

    // for every Particle xj in the neighbor hood of xi
    for (int j = 0; j < particleList[threadID].numNeighbors; j++) {
        int index = particleList[threadID].neighborIndices[j];
        float pressureTerm = (particleList[threadID].getPressure() + particleList[index].getPressure()) / (2 * particleList[index].getDensity());
        pressureGradient += (particleList[index].getMass() * pressureTerm * kernel->spikyKernelGradient(particleList[threadID], particleList[index]));
    }

    return -1.0f * pressureGradient;
}

__device__ glm::vec3 diffusionTerm(int threadID, Particle* particleList, float VISCOSITY, Kernel* kernel) {
    glm::vec3 diffusionLaplacian = glm::vec3(0.0f, 0.0f, 0.0f);

    // for every Particle xj in the neighbor hood of xi
    for (int j = 0; j < particleList[threadID].numNeighbors; j++) {
        int index = particleList[threadID].neighborIndices[j];
        glm::vec3 velocityTerm = (particleList[index].getVelocity() - particleList[threadID].getVelocity()) / particleList[index].getDensity();

        diffusionLaplacian += (particleList[index].getMass() * velocityTerm * kernel->viscosityKernelLaplacian(particleList[threadID], particleList[index]));
    }
    return diffusionLaplacian * VISCOSITY;
}

__global__ void setAccelerationsForParticles(Particle* particleList, int particleCount, float TENSION_ALPHA, float TENSION_THRESHOLD, float VISCOSITY, Kernel* kernel) {
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadID < particleCount) {
        glm::vec3 pressureForce = pressureGradient(threadID, particleList, kernel);
        glm::vec3 diffusionForce = diffusionTerm(threadID, particleList, VISCOSITY, kernel);
        glm::vec3 externalForce = glm::vec3(0.0, -9.8f * particleList[threadID].getDensity(), 0.0f);

        // calculate surface pressure/tension
        glm::vec3 acceleration = pressureForce + externalForce + diffusionForce;

        if (TENSION_ALPHA > 0.0) {
            float k = -1.0f * particleList[threadID].getColorFieldLaplacian() / glm::length(particleList[threadID].getSurfaceNormal());
            glm::vec3 tension = k * TENSION_ALPHA * particleList[threadID].getSurfaceNormal();
            if (length(tension) > TENSION_THRESHOLD) {
                acceleration += tension;
            }
        }

        acceleration /= particleList[threadID].getDensity();
        particleList[threadID].setAcceleration(acceleration);
    }
}

void setAccelerationsForParticles_CUDA(Particle* particleList, int particleCount, float TENSION_ALPHA, float TENSION_THRESHOLD, float VISCOSITY, Kernel* kernel) {
    // Calculate blocksize and grid size
    int blockSize = 512;
    int gridSize = (particleCount + blockSize - 1) / blockSize;

    // Launch CUDA kernel.
    setAccelerationsForParticles << <gridSize, blockSize >> > (particleList, particleCount, TENSION_ALPHA, TENSION_THRESHOLD, VISCOSITY, kernel);
}