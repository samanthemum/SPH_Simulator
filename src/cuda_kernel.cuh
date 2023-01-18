#include "Kernel.h"
#include "cyVector.h"

// wrapper functions to run simulation code on the GPU via CUDA
void setDensitiesForParticles_CUDA(Particle* particleList, int particleCount, Kernel* kernel);
void setSurfaceNormalFieldForParticles_CUDA(Particle* particleList, int particleCount, Kernel* kernel);
void setColorFieldLaplaciansForParticles_CUDA(Particle* particleList, int particleCount, Kernel* kernel);
void setPressuresForParticles_CUDA(Particle* particleList, int particleCount, float STIFFNESS_PARAM, float DENSITY_0_GUESS, Kernel* kernel);
void setAccelerationsForParticles_CUDA(Particle* particleList, int particleCount, float TENSION_ALPHA, float TENSION_THRESHOLD, float VISCOSITY, Kernel* kernel);
void updatePositionsAndVelocities_CUDA(Particle* particleList, cy::Vec3f* particlePositions, int particleCount, float timestep, Plane* surfaces, int numSurfaces, float ELASTICITY, float FRICTION, Kernel* kernel);