#include <cuda_runtime.h>
#include "cuda_kernel.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


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
            int index = particleList[threadID].device_neighborIndices[j];
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
            int index = particleList[threadID].device_neighborIndices[j];

            float outside_term = particleList[index].getMass() * 1 / particleList[index].getDensity();
            surfaceField += (outside_term * kernel->polyKernelGradient(particleList[threadID], particleList[index]));
            if (surfaceField.x != surfaceField.x) {
                printf("The threadID is %i\n", threadID);
                printf("The current neighbor is %i\n", index);
                printf("The current neighbor is a matchpoint %i\n", particleList[index].getIsMatchPoint());
            }
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
            int index = particleList[threadID].device_neighborIndices[j];
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
        int index = particleList[threadID].device_neighborIndices[j];
        float pressureTerm = (particleList[threadID].getPressure() + particleList[index].getPressure()) / (2 * particleList[index].getDensity());
        pressureGradient += (particleList[index].getMass() * pressureTerm * kernel->spikyKernelGradient(particleList[threadID], particleList[index]));
    }

    return -1.0f * pressureGradient;
}

__device__ glm::vec3 diffusionTerm(int threadID, Particle* particleList, float VISCOSITY, Kernel* kernel) {
    glm::vec3 diffusionLaplacian = glm::vec3(0.0f, 0.0f, 0.0f);

    // for every Particle xj in the neighbor hood of xi
    for (int j = 0; j < particleList[threadID].numNeighbors; j++) {
        int index = particleList[threadID].device_neighborIndices[j];
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

__global__ void updatePositionsAndVelocities(Particle* particleList, cy::Vec3f* particlePositions, int particleCount, float timestep, Plane* surfaces, int numSurfaces, float ELASTICITY, float FRICTION, Kernel* kernel) {
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadID < particleCount) {
        float timeStepRemaining = timestep;
        glm::vec3 acceleration = particleList[threadID].getAcceleration();
        glm::vec3 halfPointVelocity = particleList[threadID].getVelocity() + acceleration * (timeStepRemaining / 2.f);
        glm::vec3 newPosition = particleList[threadID].getPosition() + particleList[threadID].getVelocity() * timeStepRemaining + .5f * acceleration * powf(timeStepRemaining, 2.f);
        glm::vec3 newVelocity = halfPointVelocity + particleList[threadID].getAcceleration() * timeStepRemaining;

        for (int i = 0; i < numSurfaces; i++) {
            if (Particle::willCollideWithPlane(particleList[threadID].getPosition(), newPosition, particleList[threadID].getRadius(), surfaces[i])) {
                // collision stuff
                glm::vec3 velocityNormalBefore = glm::dot(newVelocity, surfaces[i].getNormal()) * surfaces[i].getNormal();
                glm::vec3 velocityTangentBefore = newVelocity - velocityNormalBefore;
                glm::vec3 velocityNormalAfter = -1 * ELASTICITY * velocityNormalBefore;
                float frictionMultiplier = glm::min((1 - FRICTION) * glm::length(velocityNormalBefore), glm::length(velocityTangentBefore));
                glm::vec3 velocityTangentAfter;
                if (glm::length(velocityTangentBefore) == 0) {
                    velocityTangentAfter = velocityTangentBefore;
                }
                else {
                    velocityTangentAfter = velocityTangentBefore - frictionMultiplier * glm::normalize(velocityTangentBefore);
                }

                newVelocity = velocityNormalAfter + velocityTangentAfter;
                float distance = particleList[threadID].getDistanceFromPlane(newPosition, particleList[threadID].getRadius(), surfaces[i]);
                glm::vec3 addedVector = glm::vec3(surfaces[i].getNormal()) * (distance * (1 + ELASTICITY));
                newPosition = newPosition + addedVector;
                // particleList[i].setPosition(newPosition);
            }
        }

        particleList[threadID].setVelocity(newVelocity);
        particleList[threadID].setPosition(newPosition);

        particlePositions[threadID].x = newPosition.x;
        particlePositions[threadID].y = newPosition.y;
        particlePositions[threadID].z = newPosition.z;
    }
}

void updatePositionsAndVelocities_CUDA(Particle* particleList, cy::Vec3f* particlePositions, int particleCount, float timestep, Plane* surfaces, int numSurfaces, float ELASTICITY, float FRICTION, Kernel* kernel) {
    // Calculate blocksize and grid size
    int blockSize = 512;
    int gridSize = (particleCount + blockSize - 1) / blockSize;

    updatePositionsAndVelocities << <gridSize, blockSize >> > (particleList, particlePositions, particleCount, timestep, surfaces, numSurfaces, ELASTICITY, FRICTION, kernel);
}