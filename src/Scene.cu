#include "Scene.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

// Create/destroy class
Scene::Scene() {
	type = SceneType::DEFAULT;
	initSceneOriginal(1.f, 1000, 5);
}

Scene::Scene(Scene::SceneCreationArgument& args) {
	cudaMallocManaged(reinterpret_cast<void**>(&surfaces), sizeof(Plane) * 6);
	switch (args.type) {
	case SceneType::DAM_BREAK:
		initSceneDamBreak(args.resolutionScale, args.particleCount, args.maxMatchpoints);
		break;
	case SceneType::DROP:
		initSceneDrop(args.maxMatchpoints, args.maxRadius, args.sphereRadius, args.shape);
		break;
	case SceneType::SPLASH:
		initSceneSplash(args.resolutionScale, args.particleCount, args.maxMatchpoints, args.maxRadius, args.sphereRadius, args.shape);
		break;
	default:
		initSceneOriginal(args.particleScale, args.particleCount, args.maxMatchpoints);
	}
}

Scene::~Scene() {
	freeCudaMemory();
}

// Setters
void Scene::setParticleMass(float mass) {
	particleMass = mass;
}

void Scene::setParticleRadius(float radius) {
	particleRadius = radius;
}

void Scene::setParticleCount(int newCount) {
	this->previousParticleCount = particleCount;
	this->particleCount = newCount;
}

// Getters
Particle* Scene::getParticleList() {
	return particleList;
}

Plane* Scene::getSurfaces() {
	return surfaces;
}

cy::Vec3f* Scene::getParticlePositions() {
	return particlePositions;
}

int Scene::getSurfaceCount() {
	return numSurfaces;
}

int Scene::getParticleCount() {
	return particleCount;
}

int Scene::getPreviousParticleCount() {
	return previousParticleCount;
}

// Memory helpers
void Scene::freeCudaMemory() {
	gpuErrchk(cudaDeviceSynchronize());
	std::cout << "Freeing scene memory" << std::endl;
	if (particleList != nullptr) {
		std::cout << "Particle list exists!" << std::endl;
		for (int i = 0; i < previousParticleCount; i++) {
			if (particleList[i].neighborIndices != nullptr) {
				cudaFreeHost(particleList[i].neighborIndices);
				particleList[i].neighborIndices = nullptr;
			}
		}
		gpuErrchk(cudaDeviceSynchronize());
		cudaFree(particleList);
		particleList = nullptr;
	}
	if (particlePositions != nullptr) {
		cudaFree(particlePositions);
	}
	particlePositions = nullptr;
	std::cout << "Freed scene memory" << std::endl;
}

// Scene creation
void Scene::initSceneOriginal(float particleScale, int particleCount, int maxMatchpoints) {
	initParticleList(particleCount, maxMatchpoints);

	Plane ground(glm::vec3(0.0, 1.0, 0.0), glm::vec3(0.0, -2.0 * particleScale, 0.0));
	Plane wall_1(glm::vec3(0.0f, 0.0, -1.0), glm::vec3(0.0, 0.0, 21.0f));
	Plane wall_2(glm::vec3(1.0, 0.0, .0), glm::vec3(-1.0, 0.0, 0.0));
	Plane wall_3(glm::vec3(0.0, 0.0, 1.0), glm::vec3(0.0, 0.0, -1.0));
	Plane wall_4(glm::vec3(-1.0, 0.0, 0.0), glm::vec3(21.0f, 0.0, 0.0));

	surfaces[0] = ground;
	surfaces[1] = wall_1;
	surfaces[2] = wall_2;
	surfaces[3] = wall_3;
	surfaces[4] = wall_4;
	numSurfaces = 5;

}

void Scene::initSceneDamBreak(float resolutionScale, int particleCount, int maxMatchpoints) {
	initParticleListAtRest(particleCount, maxMatchpoints, resolutionScale);

	Plane ground(glm::vec3(0.0, 1.0, 0.0), glm::vec3(0.0, 0.0f - .5, 0.0));
	Plane wall_1(glm::vec3(0.0f, 0.0, -1.0), glm::vec3(0.0, 0.0, 20.0f + .5));
	Plane wall_2(glm::vec3(1.0, 0.0, .0), glm::vec3(0.0 - .5, 0.0f, 0.0));
	Plane wall_3(glm::vec3(0.0, 0.0, 1.0), glm::vec3(0.0, 0.0, 0.0 - .5));
	Plane wall_5(glm::vec3(-1.0, 0.0, 0.0), glm::vec3(40 + .5, 0.0, 0.0));
	Plane wall_4(glm::vec3(-1.0, 0.0, 0.0), glm::vec3(20 + .5, 0.0, 0.0));

	surfaces[0] = ground;
	surfaces[1] = wall_1;
	surfaces[2] = wall_2;
	surfaces[3] = wall_3;
	surfaces[4] = wall_5;
	surfaces[5] = wall_4;
	numSurfaces = 6;
}

void Scene::initSceneSplash(float resolutionScale, int particleCount, int maxMatchpoints, float maxRadius, float sphereRadius, std::shared_ptr<Shape> shape) {
	initParticleListAtRest(particleCount, maxMatchpoints, resolutionScale);
	initParticleShape(maxRadius, maxMatchpoints, sphereRadius, shape);

	Plane ground(glm::vec3(0.0, 1.0, 0.0), glm::vec3(0.0, 0.0f - .5f, 0.0));
	Plane wall_1(glm::vec3(0.0f, 0.0, -1.0), glm::vec3(0.0, 0.0, 20.0f + .5f));
	Plane wall_2(glm::vec3(1.0, 0.0, .0), glm::vec3(0.0 - .5f, 0.0f, 0.0));
	Plane wall_3(glm::vec3(0.0, 0.0, 1.0), glm::vec3(0.0, 0.0, 0.0 - .5f));
	Plane wall_4(glm::vec3(-1.0, 0.0, 0.0), glm::vec3(20 + .5f, 0.0, 0.0));

	// initialize surfaces
	surfaces[0] = ground;
	surfaces[1] = wall_1;
	surfaces[2] = wall_2;
	surfaces[3] = wall_3;
	surfaces[4] = wall_4;
	numSurfaces = 5;
}

void Scene::initSceneDrop(int maxMatchpoints, float maxRadius, float sphereRadius, std::shared_ptr<Shape> shape) {
	setParticleCount(0);
	initParticleShape(maxRadius, maxMatchpoints, sphereRadius, shape);

	Plane ground(glm::vec3(0.0, 1.0, 0.0), glm::vec3(0.0, 0.0f - .5f, 0.0));
	Plane wall_1(glm::vec3(0.0f, 0.0, -1.0), glm::vec3(0.0, 0.0, 20.0f + .5f));
	Plane wall_2(glm::vec3(1.0, 0.0, .0), glm::vec3(0.0 - .5f, 0.0f, 0.0));
	Plane wall_3(glm::vec3(0.0, 0.0, 1.0), glm::vec3(0.0, 0.0, 0.0 - .5f));
	Plane wall_4(glm::vec3(-1.0, 0.0, 0.0), glm::vec3(20 + .5f, 0.0, 0.0));

	surfaces[0] = ground;
	surfaces[1] = wall_1;
	surfaces[2] = wall_2;
	surfaces[3] = wall_3;
	surfaces[4] = wall_4;
	numSurfaces = 5;
}

// Scene creation helpers
void Scene::initParticleListAtRest(int particleCount, int maxMatchpoints, float resolutionScale) {
	// Free up old memory if it was used
	freeCudaMemory();

	// Allocate new memory
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMallocManaged(reinterpret_cast<void**>(&particleList), ((particleCount + maxMatchpoints) * sizeof(Particle))));
	gpuErrchk(cudaMallocManaged(reinterpret_cast<void**>(&particlePositions), ((particleCount + maxMatchpoints) * sizeof(cy::Vec3f))));
	gpuErrchk(cudaDeviceSynchronize());

	// put them in a cube-ish shape for ease of access
	float scaleFactor = (powf(resolutionScale, (1.f / 3.f)) / powf(particleCount, (1.f / 3.f)));
	std::cout << "The scale factor is " << scaleFactor << std::endl;
	int depth = 20.0f * (1.0f / scaleFactor);
	std::cout << "The depth is " << depth << std::endl;
	int slice = roundf((float)particleCount / depth);
	std::cout << "Value of slice: " << slice << std::endl;
	int width = roundf((float)slice / (float)depth);
	std::cout << "The width is " << width << std::endl;
	int height = roundf((float)slice / (float)width);
	std::cout << "The height is " << height << std::endl;

	float volume = (20 * 20 * 20);
	float volumePerParticle = volume / particleCount;
	std::cout << "The particle count is " << particleCount << std::endl;

	// Update particle count
	setParticleCount(particleCount);

	// create a uniform random distribution
	std::uniform_real_distribution<float> distribution(0.0f, 20.0f);
	std::default_random_engine generator;

	for (int i = 0; i < depth; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < height; k++) {
				Particle p;

				// generate particle positions from distribution
				float x_position = ((float)distribution(generator));
				float y_position = ((float)distribution(generator) * .4);
				float z_position = ((float)distribution(generator));
				if (k % 2 == 1) {
					x_position += (.5 * resolutionScale);
				}

				// initialize particle values
				p.setPosition(glm::vec3(x_position, y_position, z_position));
				//p.setDensity(DENSITY_0_GUESS);
				p.setMass(particleMass);
				p.setVelocity(glm::vec3(0.0f, 0.0f, 0.0f));
				p.setRadius(particleRadius);

				int index = (slice * i) + (height * j) + k;
				if (index >= particleCount) {
					std::cout << "value of i: " << i << std::endl;
					std::cout << "Index too large: " << index << std::endl;
				}
				particleList[index] = p;
				particlePositions[index] = cy::Vec3f(x_position, y_position, z_position);

			}
		}
	}
	std::cout << "Finished particle initialization" << std::endl;
}

void Scene::initParticleShape(float maxRadius, int maxMatchpoints, float sphereRadius, std::shared_ptr<Shape> shape) {
	// sample the sphere for particle positions
	std::vector<Eigen::Matrix<float, 3, 1>> meshParticles = shape->sampleMesh(maxRadius / (sphereRadius * 2.0f));
	int usedParticles = meshParticles.size() - (meshParticles.size() % 10);

	// update the size of particles
	Particle* shapeParticles;
	gpuErrchk(cudaMallocManaged(reinterpret_cast<void**>(&shapeParticles), ((particleCount + usedParticles + maxMatchpoints) * sizeof(Particle))));
	cy::Vec3f* newPositions;
	gpuErrchk(cudaMallocManaged(reinterpret_cast<void**>(&newPositions), ((particleCount + usedParticles + maxMatchpoints) * sizeof(Particle))));
	gpuErrchk(cudaDeviceSynchronize());

	// copy over old information
	for (int i = 0; i < particleCount; i++) {
		shapeParticles[i] = particleList[i];
		newPositions[i] = particlePositions[i];
	}

	// free old memory
	freeCudaMemory();

	// approximate density of the sphere
	float volume = (4.0 / 3.0) * M_PI * powf(1.75f, 3.0f);
	float density_estimate = (particleMass * usedParticles) / volume;
	for (int i = 0; i < usedParticles; i++) {
		//		// translate x, y, and z
		float x = meshParticles.at(i)[0];
		float y = meshParticles.at(i)[1];
		float z = meshParticles.at(i)[2];

		x = sphereRadius * x + 10;
		y = sphereRadius * y + 40;
		z = sphereRadius * z + 10;

		Particle p;
		p.setPosition(glm::vec3(x, y, z));
		p.setDensity(density_estimate);
		p.setMass(particleMass);
		p.setVelocity(glm::vec3(0.0f, 0.0f, 0.0f));
		p.setRadius(particleRadius);

		cy::Vec3f potentialParticle;
		potentialParticle.x = x;
		potentialParticle.y = y;
		potentialParticle.z = z;

		shapeParticles[particleCount + i] = p;
		newPositions[particleCount + i] = potentialParticle;
	}
	gpuErrchk(cudaDeviceSynchronize());
	particleList = shapeParticles;
	particlePositions = newPositions;
	this->particleCount += usedParticles;
}

void Scene::initParticleList(int particleCount, int maxMatchpoints) {
	// free old memory
	freeCudaMemory();

	// Allocate new memory
	gpuErrchk(cudaDeviceSynchronize());
	cudaMallocManaged(reinterpret_cast<void**>(&particleList), ((particleCount + maxMatchpoints) * sizeof(Particle)));
	cudaMallocManaged(reinterpret_cast<void**>(&particlePositions), ((particleCount + maxMatchpoints) * sizeof(cy::Vec3f)));
	gpuErrchk(cudaDeviceSynchronize());

	// Update particle count
	setParticleCount(particleCount);

	// put them in a cube shape for ease of access
	float depth = 20.0f;
	int slice = particleCount / depth;
	int width = slice / 20.0f;
	int height = slice / width;

	float volume = (height * width * depth);
	float volumePerParticle = volume / particleCount;

	// Arrange particles in a cube
	for (int i = 0; i < depth; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < height; k++) {
				Particle p;
				float x = j; // +(2 * scaleParticles.x - 1);
				float y = k;
				float z = i; // +(2 * scaleParticles.z - 1);
				p.setPosition(glm::vec3(x, y, z));
				//p.setDensity(DENSITY_0_GUESS);
				p.setMass(particleMass);
				p.setVelocity(glm::vec3(0.0f, 0.0f, 0.0f));
				p.setRadius(particleRadius);

				int index = (slice * i) + (height * j) + k;
				particleList[index] = p;
				particlePositions[index] = cy::Vec3f(x, y, z);

			}
		}
	}
}