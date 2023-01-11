#include "Kernel.h"

void test_kernel(double* A, double* B, double* C, int arraySize);
void setDensitiesForParticles_CUDA(Particle* particleList, int particleCount, Kernel* kernel);