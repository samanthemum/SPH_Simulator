#include "Particle.h"
#include "Kernel.h"

void kernel(double* A, double* B, double* C, int arraySize);
void setDensitiesForParticles_CUDA(Particle* particleList, int particleCount);