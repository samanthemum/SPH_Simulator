#include <cuda_runtime.h>
#include "cuda_kernel.cuh"


// calculate the density for a given particle using the given kernel
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

// wrapper function fo calculating and updating densities
void setDensitiesForParticles_CUDA(Particle* particleList, int particleCount, Kernel* kernel) {

    // Calculate blocksize and grid size
    int blockSize = 512;
    int gridSize = (particleCount + blockSize - 1) / blockSize;

    // Launch CUDA kernel.
    setDensitiesForParticles << <gridSize, blockSize >> > (particleList, particleCount, kernel);
}

// function to calculate and set the surface normal of a particle with the given kernel
__global__ void surfaceNormalField(Particle* particleList, int particleCount, Kernel* kernel) {
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (threadID < particleCount) {
        // for every Particle xj in the neighbor hood of xi
        glm::vec3 surfaceField = glm::vec3(0.0f, 0.0f, 0.0f);
        for (int j = 0; j < particleList[threadID].numNeighbors; j++) {
            int index = particleList[threadID].device_neighborIndices[j];

            float outside_term = particleList[index].getMass() * 1 / particleList[index].getDensity();
            surfaceField += (outside_term * kernel->polyKernelGradient(particleList[threadID], particleList[index]));
        }
       
        particleList[threadID].setSurfaceNormal(surfaceField);
        
    }
    
}

// wrapper function to calculate/set surface normals on GPU
void setSurfaceNormalFieldForParticles_CUDA(Particle* particleList, int particleCount, Kernel* kernel) {

    // Calculate blocksize and grid size
    int blockSize = 512;
    int gridSize = (particleCount + blockSize - 1) / blockSize;

    // Launch CUDA kernel.
    surfaceNormalField << <gridSize, blockSize >> > (particleList, particleCount, kernel);
}

// function to calculate and set the color field of a particle with the given kernel
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

// wrapper function to calculate and set the surface normal of a particle on the GPU
void setColorFieldLaplaciansForParticles_CUDA(Particle* particleList, int particleCount, Kernel* kernel) {

    // Calculate blocksize and grid size
    int blockSize = 512;
    int gridSize = (particleCount + blockSize - 1) / blockSize;

    // Launch CUDA kernel.
    colorFieldLaplacian << <gridSize, blockSize >> > (particleList, particleCount, kernel);
}

// function to calculate and set the pressure of a particle with the given kernel
__global__ void calculatePressureForParticle(Particle* particleList, int particleCount, float STIFFNESS_PARAM, float DENSITY_0_GUESS, Kernel* kernel) {
    
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadID < particleCount) {
        float pressure = STIFFNESS_PARAM * (particleList[threadID].getDensity() - DENSITY_0_GUESS);
        particleList[threadID].setPressure(pressure);
    }
}

// wrapper function to allow the CPU to call the GPU to perform pressure calculations
void setPressuresForParticles_CUDA(Particle* particleList, int particleCount, float STIFFNESS_PARAM, float DENSITY_0_GUESS, Kernel* kernel) {
    // Calculate blocksize and grid size
    int blockSize = 512;
    int gridSize = (particleCount + blockSize - 1) / blockSize;

    // Launch CUDA kernel.
    calculatePressureForParticle << <gridSize, blockSize >> > (particleList, particleCount, STIFFNESS_PARAM, DENSITY_0_GUESS, kernel);
}

// function to calculate and set the pressure gradient of a particle with the given kernel
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

// function to calculate and set the diffusion term of a particle with the given kernel
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

// function to calculate and set the accleration of a particle with the given kernel
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

// wrapper function that allows the CPU to utilize the GPU for acceleration calculations
void setAccelerationsForParticles_CUDA(Particle* particleList, int particleCount, float TENSION_ALPHA, float TENSION_THRESHOLD, float VISCOSITY, Kernel* kernel) {
    // Calculate blocksize and grid size
    int blockSize = 512;
    int gridSize = (particleCount + blockSize - 1) / blockSize;

    // Launch CUDA kernel.
    setAccelerationsForParticles << <gridSize, blockSize >> > (particleList, particleCount, TENSION_ALPHA, TENSION_THRESHOLD, VISCOSITY, kernel);
}

// function to calculate and set the positions and velocities of a particle using leapfrog integration
__global__ void updatePositionsAndVelocities(Particle* particleList, cy::Vec3f* particlePositions, int particleCount, float timestep, Plane* surfaces, int numSurfaces, float ELASTICITY, float FRICTION, Kernel* kernel) {
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadID < particleCount) {
        float timeStepRemaining = timestep;
        glm::vec3 acceleration = particleList[threadID].getAcceleration();
        glm::vec3 halfPointVelocity = particleList[threadID].getVelocity() + acceleration * (timeStepRemaining / 2.f);
        glm::vec3 newPosition = particleList[threadID].getPosition() + particleList[threadID].getVelocity() * timeStepRemaining + .5f * acceleration * powf(timeStepRemaining, 2.f);
        glm::vec3 newVelocity = halfPointVelocity + particleList[threadID].getAcceleration() * (timeStepRemaining / 2.f);

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

        if (newVelocity.x != newVelocity.x) {
            glm::vec3 pressureForce = pressureGradient(threadID, particleList, kernel);

            printf("Velocity: the current thread is particle %i\n", threadID);
            printf("Velocity: the particle pressure force was %f %f %f \n", pressureForce.x, pressureForce.y, pressureForce.z);
            printf("Velocity: the particle acceleration was %f %f %f \n", particleList[threadID].getAcceleration().x, particleList[threadID].getAcceleration().y, particleList[threadID].getAcceleration().z);
            printf("Velocity: the particle velocity was %f %f %f \n", particleList[threadID].getVelocity().x, particleList[threadID].getVelocity().y, particleList[threadID].getVelocity().z);
            printf("Velocity: the particle position was %f %f %f \n", particleList[threadID].getPosition().x, particleList[threadID].getPosition().y, particleList[threadID].getPosition().z);
        }

        particleList[threadID].setVelocity(newVelocity);
        particleList[threadID].setPosition(newPosition);

        
        

        particlePositions[threadID].x = newPosition.x;
        particlePositions[threadID].y = newPosition.y;
        particlePositions[threadID].z = newPosition.z;
    }
}

// wrapper function that allows the CPU to call on the GPU to perform acceleration integration
void updatePositionsAndVelocities_CUDA(Particle* particleList, cy::Vec3f* particlePositions, int particleCount, float timestep, Plane* surfaces, int numSurfaces, float ELASTICITY, float FRICTION, Kernel* kernel) {
    // Calculate blocksize and grid size
    int blockSize = 512;
    int gridSize = (particleCount + blockSize - 1) / blockSize;

    updatePositionsAndVelocities << <gridSize, blockSize >> > (particleList, particlePositions, particleCount, timestep, surfaces, numSurfaces, ELASTICITY, FRICTION, kernel);
}