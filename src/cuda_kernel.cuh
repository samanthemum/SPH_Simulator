#include "Kernel.h"

void test_kernel(double* A, double* B, double* C, int arraySize);
void setDensitiesForParticles_CUDA(Particle* particleList, int particleCount, Kernel* kernel);
void setSurfaceNormalFieldForParticles_CUDA(Particle* particleList, int particleCount, Kernel* kernel);
void setColorFieldLaplaciansForParticles_CUDA(Particle* particleList, int particleCount, Kernel* kernel);
void setPressuresForParticles_CUDA(Particle* particleList, int particleCount, float STIFFNESS_PARAM, float DENSITY_0_GUESS, Kernel* kernel);
void setAccelerationsForParticles_CUDA(Particle* particleList, int particleCount, float TENSION_ALPHA, float TENSION_THRESHOLD, float VISCOSITY, Kernel* kernel);