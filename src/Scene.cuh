#pragma once
#ifndef SCENE_H
#define SCENE_H

#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h>

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

// CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cuda_kernel.cuh"

// STD
#include <iostream>
#include <vector>
#include <random>
#include <stdio.h>

#include "Particle.h"
#include "cyVector.h"

#include "../../Leaven/lib/src/volumeSampler.h"
#include "../../Leaven/lib/src/surfaceSampler.h"
#include "Shape.h"

enum class SceneType {
	DEFAULT,
	DAM_BREAK,
	SPLASH,
	DROP
};

class Scene {
private:
	// Particles
	Particle* particleList = nullptr;
	cy::Vec3f* particlePositions = nullptr;
	int particleCount;
	int previousParticleCount;
	float particleMass;
	float particleRadius;

	// Surfaces
	Plane* surfaces;
	int numSurfaces;

	// Scene Type
	SceneType type;

public:
	struct SceneCreationArgument {
		SceneType type;
		int particleCount;
		int maxMatchpoints;
		float resolutionScale;
		float particleScale;
		float sphereRadius;
		float maxRadius;
		std::shared_ptr<Shape> shape;
	};
	// Create/destroy class
	Scene();
	Scene(SceneCreationArgument& args);
	~Scene();

	// Setters
	void setParticleMass(float mass);
	void setParticleRadius(float radius);
	void setParticleCount(int newCount);

	// Getters
	Particle* getParticleList();
	Plane* getSurfaces();
	cy::Vec3f* getParticlePositions();
	int getParticleCount();
	int getPreviousParticleCount();
	int getSurfaceCount();

	// Memory helpers
	void freeCudaMemory();

	// Scene creation
	void initSceneOriginal(float particleScale, int particleCount, int maxMatchpoints);
	void initSceneDamBreak(float resolutionScale, int particleCount, int maxMatchpoints);
	void initSceneSplash(float resolutionScale, int particleCount, int maxMatchpoints, float maxRadius, float sphereRadius, std::shared_ptr<Shape> shape);
	void initSceneDrop(int maxMatchpoints, float maxRadius, float sphereRadius, std::shared_ptr<Shape> shape);

	// Scene creation helpers
	void initParticleListAtRest(int particleCount, int maxMatchpoints, float resolutionScale);
	void initParticleShape(float maxRadius, int maxMatchpoints, float sphereRadius, std::shared_ptr<Shape> shape);
	void initParticleList(int particleCount, int maxMatchpoints);
};

#endif