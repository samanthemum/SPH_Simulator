#include <iostream>
#include <vector>
#include <random>
#include <stdio.h>

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

// CY classes
#include "cyPointCloud.h"
#include "cyVector.h"
#include "cyBVH.h"

#include "GLSL.h"
#include "Program.h"
#include "MatrixStack.h"
#include "Particle.h"
#include "Shape.h"
#include "Plane.h"

// GUI Creation
#include "imgui-master/imgui.h"
#include "imgui-master/backends/imgui_impl_glfw.h"
#include "imgui-master/backends/imgui_impl_opengl3.h"

// Creating PNGs
#include "Kernel.h"
#include <thread>

// Volume Sampling
#include "../../Leaven/lib/src/volumeSampler.h"
#include "../../Leaven/lib/src/surfaceSampler.h"

// CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
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

using namespace std;
using cy::Vec3f;

// Scene information
enum class Scene {
	DEFAULT,
	DAM_BREAK,
	SPLASH,
	DROP
};
Scene selected_scene = Scene::SPLASH;

GLFWwindow* window; // Main application window
string RESOURCE_DIR = "..\\resources\\"; // Where the resources are loaded from
shared_ptr<Program> prog;

bool keyToggles[256] = { false }; // only for English keyboards!
glm::vec2 cameraRotations(0, 0);
glm::vec2 mousePrev(-1, -1);

// Kernel stuff
Kernel* kernel;

// Simulation Initialization
float resolutionConstant = 8000;
uint32_t LOW_RES_COUNT = 8000;
uint32_t MID_RES_COUNT = 27000;
uint32_t HIGH_RES_COUNT = 64000;
uint32_t LOW_RES_COUNT_SHAPE = 250;
uint32_t MID_RES_COUNT_SHAPE = 1000;
uint32_t HIGH_RES_COUNT_SHAPE = 2000;
int particleCount = LOW_RES_COUNT;
int previousParticleCount = LOW_RES_COUNT;
int particleForShape = LOW_RES_COUNT_SHAPE;
float LOW_RES_RADIUS = 1.0f;
float MID_RES_RADIUS = .5f;
float HIGH_RES_RADIUS = (1.f / 3.f);
glm::vec3 scaleStructure = glm::vec3(.05f, .05f, .05f);
glm::vec3 scaleParticles = glm::vec3(.5f * (resolutionConstant / particleCount), .5f * (resolutionConstant / particleCount), .5f * (resolutionConstant / particleCount));
Particle* particleList;
Vec3f* particlePositions;

// Simulation Fluid Constants
float MAX_RADIUS = LOW_RES_RADIUS;
float SMOOTHING_RADIUS = LOW_RES_RADIUS;
float VISCOSITY = .1f;
float TIMESTEP = .025f;
float MASS = 1.0f;
float DENSITY_0_GUESS = .1f; // density of water= 1 g/cm^3
float STIFFNESS_PARAM = 60.0f;
float Y_PARAM = 7.0f;

// Collision information
Plane* surfaces;
int numSurfaces;
float FRICTION = .1f;
float ELASTICITY = .7f;

// surface tension stuff
float TENSION_ALPHA = .25f;
float TENSION_THRESHOLD = 1.0f;
float totalTime = 0.0f;


// matchpoint and keyframe system
struct Keyframe {
	std::vector<Particle> matchpoints;
	float time;
};
vector<Keyframe> keyframes;
vector<Particle> defaultMatchpoints;
int matchpointNumber = 10;
const int maxMatchpoints = 50;
unsigned int nextKeyframe = 0;
const float permittedError = .01f;
const int minIterations = 100;
float matchPointPosition[3];
float matchPointRadius;
bool CONTROL = true;

// Kd tree and shape
int steps = 0;
int steps_per_update = 3;
shared_ptr<cy::PointCloud<Vec3f, float, 3>> kdTree;
shared_ptr<Shape> lowResSphere;


// Threading information
const int N_THREADS = 10;
int PARTICLES_PER_THREAD = LOW_RES_COUNT / N_THREADS;
std::thread threads[N_THREADS];

// Recording information
int width = 1280;
int height = 960;
const char* cmd = "\"C:\\Users\\Sam Hallam\\Desktop\\Art Stuff\\ffmpeg-2021-11-10-git-44c65c6cc0-essentials_build\\bin\\ffmpeg\" -r 30 -f rawvideo -pix_fmt rgba -s 1280x960 -i - "
"-threads 0 -preset fast -y -pix_fmt yuv420p -crf 21 -vf vflip output.mp4";
FILE* ffmpeg = _popen(cmd, "wb");
int* buffer = new int[width * height];
bool recording = false;
bool recording_low_res = false;
float timePassed = 0.0f;
float end_time = 0.0f;

static void error_callback(int error, const char* description)
{
	cerr << description << endl;
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
}

static void char_callback(GLFWwindow* window, unsigned int key)
{
	keyToggles[key] = !keyToggles[key];
}

static void cursor_position_callback(GLFWwindow* window, double xmouse, double ymouse)
{
	if (mousePrev.x >= 0) {
		glm::vec2 mouseCurr(xmouse, ymouse);
		cameraRotations += 0.01f * (mouseCurr - mousePrev);
		mousePrev = mouseCurr;
	}
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	// Get the current mouse position.
	double xmouse, ymouse;
	glfwGetCursorPos(window, &xmouse, &ymouse);
	// Get current window size.
	int width, height;
	glfwGetWindowSize(window, &width, &height);
	if (action == GLFW_PRESS) {
		mousePrev.x = xmouse;
		mousePrev.y = ymouse;
	}
	else {
		mousePrev[0] = -1;
		mousePrev[1] = -1;
	}
}

// Frees memory allocated through CUDA functions
void freeCudaMemory() {
	if (particleList != nullptr) {
		for (int i = 0; i < previousParticleCount + matchpointNumber; i++) {
			if (particleList[i].neighborIndices != nullptr) {
				cudaFreeHost(particleList[i].neighborIndices);
			}
		}
		gpuErrchk(cudaDeviceSynchronize());
		cudaFree(particleList);
	}
	if (particlePositions != nullptr) {
		cudaFree(particlePositions);
	}
}

// initialize particle list so that it starts at rest
void initParticleListAtRest() {

	// Free up old memory if it was used
	freeCudaMemory();

	// Allocate new memory
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMallocManaged(reinterpret_cast<void**>(&particleList), ((particleCount + maxMatchpoints) * sizeof(Particle))));
	gpuErrchk(cudaMallocManaged(reinterpret_cast<void**>(&particlePositions), ((particleCount + maxMatchpoints) * sizeof(Vec3f))));
	gpuErrchk(cudaDeviceSynchronize());

	// put them in a cube-ish shape for ease of access
	float scaleFactor = (powf(resolutionConstant, (1.f / 3.f)) / powf(particleCount, (1.f / 3.f)));
	cout << "The scale factor is " << scaleFactor << endl;
	int depth = 20.0f * (1.0f / scaleFactor);
	cout << "The depth is " << depth << endl;
	int slice = roundf((float)particleCount / depth);
	cout << "Value of slice: " << slice << endl;
	int width = roundf((float)slice / (float)depth);
	cout << "The width is " << width << endl;
	int height = roundf((float)slice / (float)width);
	cout << "The height is " << height << endl;

	float volume = (20 * 20 * 20);
	float volumePerParticle = volume / particleCount;
	cout << "The particle count is " << particleCount << endl;

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
					x_position += (.5 * scaleFactor);
				}

				// initialize particle values
				p.setPosition(glm::vec3(x_position, y_position, z_position));
				p.setDensity(DENSITY_0_GUESS);
				p.setMass(MASS);
				p.setVelocity(glm::vec3(0.0f, 0.0f, 0.0f));
				p.setRadius(scaleParticles.x);

				int index = (slice * i) + (height * j) + k;
				if (index >= particleCount) {
					cout << "value of i: " << i << endl;
					cout << "Index too large: " << index << endl;
				}
				particleList[index] = p;
				particlePositions[index] = Vec3f(x_position, y_position, z_position);

			}
		}
	}
	cout << "Finished particle initialization" << endl;
}

// initializes the particle list in a specific shape
void initParticleShape() {

	// sample the sphere for particle positions
	float sphereRadius = 5.0f;
	std::vector<Eigen::Matrix<float, 3, 1>> meshParticles = lowResSphere->sampleMesh(MAX_RADIUS / (sphereRadius * 2.0f));
	int usedParticles = meshParticles.size() - (meshParticles.size() % 10);

	// update the size of particles
	Particle* shapeParticles;
	gpuErrchk(cudaMallocManaged(reinterpret_cast<void**>(&shapeParticles), ((particleCount + usedParticles + maxMatchpoints) * sizeof(Particle))));
	Vec3f* newPositions;
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
	float density_estimate = (MASS * usedParticles) / volume;
	for (int i = 0; i < usedParticles; i++) {
		//		// translate x, y, and z
		float x = meshParticles.at(i)[0];
		float y = meshParticles.at(i)[1];
		float z = meshParticles.at(i)[2];

		x = sphereRadius * x + 10;
		y = sphereRadius * y + 70;
		z = sphereRadius * z + 10;

		Particle p;
		p.setPosition(glm::vec3(x, y, z));
		p.setDensity(density_estimate);
		p.setMass(MASS);
		p.setVelocity(glm::vec3(0.0f, 0.0f, 0.0f));
		p.setRadius(scaleParticles.x);

		Vec3f potentialParticle;
		potentialParticle.x = x;
		potentialParticle.y = y;
		potentialParticle.z = z;

		shapeParticles[particleCount + i] = p;
		newPositions[particleCount + i] = potentialParticle;
	}
	gpuErrchk(cudaDeviceSynchronize());
	particleList = shapeParticles;
	particlePositions = newPositions;
	particleCount += usedParticles;
	PARTICLES_PER_THREAD = particleCount / N_THREADS;
}

// initializes the particle list in a cube, but not at rest
void initParticleList() {
	
	// free old memory
	freeCudaMemory();

	// Allocate new memory
	gpuErrchk(cudaDeviceSynchronize());
	cudaMallocManaged(reinterpret_cast<void**>(&particleList), ((particleCount + maxMatchpoints) * sizeof(Particle)));
	cudaMallocManaged(reinterpret_cast<void**>(&particlePositions), ((particleCount + maxMatchpoints) * sizeof(Vec3f)));
	gpuErrchk(cudaDeviceSynchronize());

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
				p.setDensity(DENSITY_0_GUESS);
				p.setMass(MASS);
				p.setVelocity(glm::vec3(0.0f, 0.0f, 0.0f));
				p.setRadius(scaleParticles.x);

				int index = (slice * i) + (height * j) + k;
				particleList[index] = p;
				particlePositions[index] = Vec3f(x, y, z);

			}
		}
	}
}

// Creates the default scene: a cube of water falling into a box
void initSceneOriginal() {
	initParticleList();

	Plane ground(glm::vec3(0.0, 1.0, 0.0), glm::vec3(0.0, -2.0 * scaleParticles.x, 0.0));
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

// Creates the dam break scene
void initSceneDamBreak() {
	initParticleListAtRest();

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

// Creates the splash scene
void initSceneSplash() {
	initParticleListAtRest();
	initParticleShape();

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

// Creates the drop scene
void initSceneDrop() {
	particleCount = 0;
	initParticleShape();

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

// initializes the kd tree for use
void initKdTree() {
	if (!kdTree) {
		cout << "Making new kd tree!" << endl;
		kdTree = make_shared<cy::PointCloud<Vec3f, float, 3>>(particleCount + matchpointNumber, particlePositions);
	}
	else {
		kdTree->Build(particleCount + matchpointNumber, particlePositions);
	}
}

// initialize the match points
void initMatchPoints() {

	// clear frames and old matchpoints
	keyframes.clear();
	defaultMatchpoints.clear();

	// create a uniform random distribution
	std::uniform_int_distribution<int> distribution(0, (particleCount - 1));
	std::default_random_engine generator;

	// for the user defined matchpoint number
	for (int i = 0; i < matchpointNumber; i++) {
		// Choose a uniform random particle index
		int particleIndex = (int)distribution(generator);

		// initialize a particle to act as a matchpoint
		Particle matchPoint;
		matchPoint.setPosition(particleList[particleIndex].getPosition());
		float radius = rand() % 3 + 1;
		matchPoint.setRadius(radius);
		matchPoint.setMass(1.0f);
		Vec3f matchpointPosition = Vec3f(matchPoint.getPosition().x, matchPoint.getPosition().y, matchPoint.getPosition().z);
		particlePositions[particleCount + i] = matchpointPosition;
		matchPoint.setIsMatchpoint(true);

		// Add it to the particle list
		defaultMatchpoints.push_back(matchPoint);
		particleList[particleCount + i] = matchPoint;
	}
}

// Set the neighbors for a particle
void setNeighbors(Particle& x, int xIndex) {

	// Get all neighbors within a radius
	// float radius = x.getIsMatchPoint() ? x.getRadius() : MAX_RADIUS;
	float radius = xIndex >= particleCount ? x.getRadius() : MAX_RADIUS;
	cy::PointCloud<Vec3f, float, 3>::PointInfo* info = new cy::PointCloud<Vec3f, float, 3>::PointInfo[Particle::maxNeighborsAllowed];
	int numPointsInRadius = kdTree->GetPoints(particlePositions[xIndex], sqrt(radius), Particle::maxNeighborsAllowed, info);

	// free old neighbors, if there
	if (x.neighborIndices != nullptr) {
		gpuErrchk(cudaFreeHost(x.neighborIndices));
		gpuErrchk(cudaDeviceSynchronize());
	}

	// allocate memory for new neighbor list
	gpuErrchk(cudaHostAlloc(reinterpret_cast<void**>(&x.neighborIndices), numPointsInRadius * sizeof(int), cudaHostAllocMapped));
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaSetDeviceFlags(cudaDeviceMapHost));
	gpuErrchk(cudaDeviceSynchronize());
	if (numPointsInRadius > 0) {
		gpuErrchk(cudaHostGetDevicePointer((void**)&x.device_neighborIndices, (void*)x.neighborIndices, 0));
		gpuErrchk(cudaDeviceSynchronize());
	}
	

	// set neighbors, ignoring matchpoints and self inclusion
	x.numNeighbors = numPointsInRadius;
	for (int i = 0; i < numPointsInRadius; i++) {
		if (xIndex != info[i].index && info[i].index < particleCount) {
			x.neighborIndices[i] = (int)info[i].index;
		}
		else {
			x.numNeighbors--;
		}
	}

	// delete point info
	delete[] info;
}

// calculates the average mass for a particle at rest
void initAverageMass() {
	// use the at rest distribution and setup tree
	initParticleListAtRest();
	initKdTree();
	for (int i = 0; i < particleCount; i++) {
		setNeighbors(particleList[i], i);
	}

	// Calculate the mass it takes for each particle to be at approximately the rest density
	float averageMass = 0.0f;
	for (int i = 0; i < particleCount; i++) {
		Particle xi = particleList[i];
		float thisKernel = kernel->polyKernelFunction(xi, xi, false);
		float kernelSum = 0.f;
		for (int j = 0; j < xi.numNeighbors; j++) {
			kernelSum += kernel->polyKernelFunction(xi, particleList[xi.neighborIndices[j]], false);
		}
		averageMass += (DENSITY_0_GUESS / (thisKernel + kernelSum) / particleCount);
	}

	// set the average mass
	cout << "Finished calculating average mass" << endl;
	MASS = averageMass;
	cout << "The average mass was " << MASS << endl;
}

// initializes everything required before simulation start
static void init()
{
	GLSL::checkVersion();

	// allocate the kernel in shared memory
	gpuErrchk(cudaMallocManaged(reinterpret_cast<void**>(&kernel), sizeof(Kernel)));
	gpuErrchk(cudaDeviceSynchronize());
	kernel->setSmoothingRadius(SMOOTHING_RADIUS);

	// initialize average mass
	initAverageMass();

	// Set background color
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	// Enable z-buffer test
	glEnable(GL_DEPTH_TEST);

	// Initialize the GLSL program.
	prog = make_shared<Program>();
	prog->setShaderNames(RESOURCE_DIR + "phong_vert.glsl", RESOURCE_DIR + "phong_frag.glsl");
	prog->setVerbose(true);
	prog->init();
	prog->addUniform("P");
	prog->addUniform("MV");
	prog->addUniform("lightPos");
	prog->addUniform("ka");
	prog->addUniform("kd");
	prog->addUniform("ks");
	prog->addUniform("s");
	prog->addAttribute("aPos");
	prog->addAttribute("aNor");
	prog->setVerbose(false);

	// Initialize time.
	glfwSetTime(0.0);
	keyToggles[(unsigned)'l'] = true;

	// initialize shape for sphere
	lowResSphere = make_shared<Shape>();
	lowResSphere->loadMesh(RESOURCE_DIR + "low_res_sphere.obj");
	lowResSphere->init();

	// initialize surfaces in shared memory
	cudaMallocManaged(reinterpret_cast<void**>(&surfaces), sizeof(Plane) * 6);

	// initialize particles and tree
	keyframes.clear();
	switch (selected_scene) {
	case Scene::DAM_BREAK:
		initSceneDamBreak();
		break;
	case Scene::SPLASH:
		initSceneSplash();
		break;
	case Scene::DROP:
		initSceneDrop();
		break;
	default:
		initSceneOriginal();
	}

	// initialize matchpoints
	initMatchPoints();

	// start kd tree and set neighbors
	initKdTree();
	for (int i = 0; i < particleCount; i++) {
		setNeighbors(particleList[i], i);
	}

	// If there were any OpenGL errors, this will print something.
	// You can intersperse this line in your code to find the exact location
	// of your OpenGL error.
	GLSL::checkError(GET_FILE_LINE);
}

// draw particle on the shared pointer
void drawParticles(shared_ptr<MatrixStack>& MV) {
	MV->pushMatrix();

	// color particles blue because water
	glUniform3f(prog->getUniform("kd"), 0.0f, 0.3f, .7f);
	glUniform3f(prog->getUniform("ka"), 0.0f, 0.3f, .7f);
	glUniform3f(prog->getUniform("ks"), 0.0f, 0.3f, .7f);
	glUniform3f(prog->getUniform("lightPos"), 20.0f, 20.0f, -20.0f);
	MV->scale(scaleStructure);

	// draw each particle
	for (int i = 0; i < particleCount; i++) {
		MV->pushMatrix();
		MV->translate(particleList[i].getPosition());
		MV->scale(scaleParticles);
		// calculate blue and green based on particle velocity
		glUniformMatrix4fv(prog->getUniform("MV"), 1, GL_FALSE, glm::value_ptr(MV->topMatrix()));
		lowResSphere->draw(prog);
		MV->popMatrix();
	}
	MV->popMatrix();
}

// draw matchpoint on the shared pointer
void drawMatchpoints(shared_ptr<MatrixStack>& MV) {
	MV->pushMatrix();

	// color particles blue because water
	glUniform3f(prog->getUniform("kd"), 1.0f, 0.0f, .0f);
	glUniform3f(prog->getUniform("ka"), 1.0f, 0.0f, .0f);
	glUniform3f(prog->getUniform("ks"), 0.0f, 0.3f, .7f);
	glUniform3f(prog->getUniform("lightPos"), 20.0f, 20.0f, -20.0f);
	MV->scale(scaleStructure);

	// draw each particle
	for (int i = 0; i < matchpointNumber; i++) {
		MV->pushMatrix();
		MV->translate(particleList[particleCount + i].getPosition());

		float radius = particleList[particleCount + i].getRadius();
		glm::vec3 matchpointScale = glm::vec3(radius, radius, radius);
		MV->scale(matchpointScale);
		// calculate blue and green based on particle velocity
		glUniformMatrix4fv(prog->getUniform("MV"), 1, GL_FALSE, glm::value_ptr(MV->topMatrix()));
		lowResSphere->draw(prog);
		MV->popMatrix();
	}
	MV->popMatrix();
}

// render a frame
void render()
{
	// Update time.
	double t = glfwGetTime();
	// Get current frame buffer size.
	glfwGetFramebufferSize(window, &width, &height);
	glViewport(0, 0, width, height);

	// Clear buffers
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	auto P = make_shared<MatrixStack>();
	auto MV = make_shared<MatrixStack>();
	P->pushMatrix();
	MV->pushMatrix();

	double aspect = (double)width / height;
	P->multMatrix(glm::ortho(-aspect, aspect, -1.0, 1.0, -2.0, 2.0));
	MV->rotate(cameraRotations.y, glm::vec3(1, 0, 0));
	MV->rotate(cameraRotations.x, glm::vec3(0, 1, 0));

	// Bind the program
	prog->bind();
	glUniformMatrix4fv(prog->getUniform("P"), 1, GL_FALSE, glm::value_ptr(P->topMatrix()));
	glUniformMatrix4fv(prog->getUniform("MV"), 1, GL_FALSE, glm::value_ptr(MV->topMatrix()));

	// draw particles
	if (keyToggles[unsigned('c')] % 2 == 0) {
		drawParticles(MV);
	}
	else {
		drawMatchpoints(MV);
	}
	

	// Unbind the program
	prog->unbind();

	// Pop matrix stacks.
	MV->popMatrix();
	P->popMatrix();

	GLSL::checkError(GET_FILE_LINE);
}

// samples the density around a given matchpoint
// uses the kernel from the original Keyser paper
float sampleDensityForMatchpoint(const Particle x) {
	float sample = 0;
	float normalizer = 0;
	for (int j = 0; j < x.numNeighbors; j++) {
		int index = x.neighborIndices[j];
		sample += (particleList[index].getDensity() * kernel->samplingKernel(x, particleList[index], true));
		normalizer += (kernel->samplingKernel(x, particleList[index], true));
	}

	return sample / normalizer;
}

// samples the velocity (direction) around a given matchpoint
// uses the same kernel from the original Keyser pape
glm::vec3 sampleVelocityForMatchpoint(const Particle x) {
	glm::vec3 sample = glm::vec3(0.f, 0.f, 0.f);
	float normalizer = 0;
	for (int j = 0; j < x.numNeighbors; j++) {
		int index = x.neighborIndices[j];
		sample += (particleList[index].getVelocity() * kernel->samplingKernel(x, particleList[index], true));
		normalizer += (kernel->samplingKernel(x, particleList[index], true));
	}

	return sample / normalizer;
}

// samples the velocity (curl) around a given matchpoint
// uses the same kernel from the original Keyser pape
glm::vec3 sampleCurlForMatchpoint(const Particle x) {
	glm::vec3 sample = glm::vec3(0.f, 0.f, 0.f);
	float normalizer = 0;
	for (int j = 0; j < x.numNeighbors; j++) {
		int index = x.neighborIndices[j];
		glm::vec3 velocity_direction = particleList[index].getVelocity() * kernel->samplingKernel(x, particleList[index], true);
		glm::vec3 cross_product = glm::cross(velocity_direction, particleList[index].getPosition() - x.getPosition());
		sample += cross_product;
		normalizer += (kernel->samplingKernel(x, particleList[index], true));
	}

	return sample / normalizer;
}


// set neighbors for all particles within the given indices
void setNeighborsForParticles(int start_index, int end_index) {
	for (int i = start_index; i < end_index; i++) {
		setNeighbors(particleList[i], i);
	}
}

// update the samples at each matchpoint
void updateMatchPoints(float time) {

	// create a new keyframe
	Keyframe k;
	k.time = time + timePassed;

	// TODO: add the matchpoints to tree and calculate each trait of match point
	for (int i = 0; i < matchpointNumber; i++) {
		particlePositions[particleCount + i] = Vec3f(defaultMatchpoints.at(i).getPosition().x, defaultMatchpoints.at(i).getPosition().y, defaultMatchpoints.at(i).getPosition().z);
		// set the neighbors for match point
		setNeighbors(defaultMatchpoints.at(i), particleCount + i);

		// calculate density for matchpoint using the sampling method
		defaultMatchpoints.at(i).setDensity(sampleDensityForMatchpoint(defaultMatchpoints.at(i)));

		// calculate velocity for matchpoint using the sampling method
		defaultMatchpoints.at(i).setVelocity(sampleVelocityForMatchpoint(defaultMatchpoints.at(i)));

		// calculate curl for matchpoint using the sampling method
		defaultMatchpoints.at(i).setCurl(sampleCurlForMatchpoint(defaultMatchpoints.at(i)));

		// add the match point to the keyframe
		k.matchpoints.push_back(defaultMatchpoints.at(i));
	}

	keyframes.push_back(k);
}

void updateFluid(float time) {

	// update the kd tree
	gpuErrchk(cudaDeviceSynchronize());
	if (steps == steps_per_update) {
		initKdTree();
		steps = 0;

		// set neighbors for all particles
		for (int i = 0; i < N_THREADS; i++) {
			threads[i] = thread(setNeighborsForParticles, i * PARTICLES_PER_THREAD, (i + 1) * PARTICLES_PER_THREAD);
		}

		for (int i = 0; i < N_THREADS; i++) {
			threads[i].join();
		}
	}
	gpuErrchk(cudaDeviceSynchronize());

	// update density
	setDensitiesForParticles_CUDA(particleList, particleCount, kernel);
	gpuErrchk(cudaDeviceSynchronize());

	// update surface normal
	setSurfaceNormalFieldForParticles_CUDA(particleList, particleCount, kernel);
	gpuErrchk(cudaDeviceSynchronize());

	// update color field laplacian
	setColorFieldLaplaciansForParticles_CUDA(particleList, particleCount, kernel);
	gpuErrchk(cudaDeviceSynchronize());

	// update the pressures
	setPressuresForParticles_CUDA(particleList, particleCount, STIFFNESS_PARAM, DENSITY_0_GUESS, kernel);
	gpuErrchk(cudaDeviceSynchronize());

	// calculate acceleration
	setAccelerationsForParticles_CUDA(particleList, particleCount, TENSION_ALPHA, TENSION_THRESHOLD, VISCOSITY, kernel);
	gpuErrchk(cudaDeviceSynchronize());

	// update positions
	updatePositionsAndVelocities_CUDA(particleList, particlePositions, particleCount, time, surfaces, numSurfaces, ELASTICITY, FRICTION, kernel);
	gpuErrchk(cudaDeviceSynchronize());

	// do key frame stuff
	if (!recording) {
		updateMatchPoints(time);
	}
	else if (CONTROL) {
		// apply match point control
		// 0. Figure out if we're at a keyframe time
		if (nextKeyframe < keyframes.size() && keyframes.at(nextKeyframe).time == (timePassed + time)) {
			// loop through matchpoints
			
			for (int i = 0; i < matchpointNumber; i++) {
				int iterations = 0;
				Particle matchpoint = keyframes.at(nextKeyframe).matchpoints.at(i);

				// 1. Sample high resolution model at point
				Particle highResSample;
				highResSample.setPosition(keyframes.at(nextKeyframe).matchpoints.at(i).getPosition());
				highResSample.setRadius(keyframes.at(nextKeyframe).matchpoints.at(i).getRadius());
				highResSample.setMass(keyframes.at(nextKeyframe).matchpoints.at(i).getMass());
				highResSample.setIsMatchpoint(true);

				// get high res sample neighbors
				particlePositions[particleCount + i] = Vec3f(highResSample.getPosition().x, highResSample.getPosition().y, highResSample.getPosition().z);
				setNeighbors(highResSample, particleCount + i);
				// cout << "Sample has " << highResSample.numNeighbors << endl;

				// calculate high res density
				highResSample.setDensity(sampleDensityForMatchpoint(highResSample));

				// calculate high res velocity
				highResSample.setVelocity(sampleVelocityForMatchpoint(highResSample));

				// calculate high res curl
				highResSample.setCurl(sampleCurlForMatchpoint(highResSample));

				// 2. Calculate error between high and low res value
				float densityError = matchpoint.getMass() * (matchpoint.getDensity() - highResSample.getDensity());
				float absError_density = abs(((matchpoint.getDensity() - highResSample.getDensity()) / matchpoint.getDensity()));

				glm::vec3 velocityError = matchpoint.getMass() * (matchpoint.getVelocity() - highResSample.getVelocity());
				float absError_velocity = length((matchpoint.getVelocity() - highResSample.getVelocity())) / length(matchpoint.getVelocity());

				glm::vec3 curlError = matchpoint.getMass() * (matchpoint.getCurl() - highResSample.getCurl());
				float absError_curl = length((matchpoint.getCurl() - highResSample.getCurl())) / length(matchpoint.getCurl());

				/*cout << "Match point mass is " << matchpoint.getMass() << endl;

				cout << "Raw Density error is " << densityError << endl;
				cout << "Initial Density error is " << absError_density << endl;
				cout << "Initial Velocity error is " << absError_velocity << endl;
				cout << "Initial Curl error is " << absError_curl << endl;*/

				while (iterations < minIterations && (absError_density > permittedError || absError_velocity > permittedError || absError_curl > permittedError)) {
					// 3. Calcuate G'(r, x)
					float totalError = 0;
					for (int j = 0; j < highResSample.numNeighbors; j++) {
						int index = highResSample.neighborIndices[j];
						float gravity_kernel_value = kernel->samplingKernel(highResSample, particleList[index], true);
						
						totalError += powf(gravity_kernel_value, 2.0f);
					}
					/*cout << "Total error denominator is " << totalError << endl;*/

					// 4. Apply control for each neighbor
					/*if (highResSample.numNeighbors == 0) {
						cout << "It has no neighbors" << endl;
					}*/
					if (totalError >= 0.00001f) {
						for (int j = 0; j < highResSample.numNeighbors; j++) {
							int index = highResSample.neighborIndices[j];
							float gravity_kernel_value = kernel->samplingKernel(highResSample, particleList[index], true);

							if (gravity_kernel_value / totalError == 0) {
								/*cout << "Function has no effect!" << endl*/;
							}


							float newDensity = particleList[index].getDensity() + (densityError * (gravity_kernel_value / totalError));
							glm::vec3 newVelocity = particleList[index].getVelocity() + velocityError * (gravity_kernel_value / totalError);
							newVelocity += (gravity_kernel_value / totalError) * glm::cross(velocityError, particleList[index].getPosition() - highResSample.getPosition());
							glm::vec3 newCurl = particleList[index].getCurl() + (gravity_kernel_value / totalError) * glm::cross(velocityError, particleList[index].getPosition() - highResSample.getPosition());

							particleList[index].setDensity(newDensity);
							particleList[index].setVelocity(newVelocity);
							particleList[index].setCurl(newCurl);
						}
					}
					else {
						for (int j = 0; j < highResSample.numNeighbors; j++) {
							int index = highResSample.neighborIndices[j];
							float gravity_kernel_value = kernel->samplingKernel(highResSample, particleList[index], true);
							/*cout << "gravity value is " << gravity_kernel_value << endl;*/
						}
						
					}
					

					// update sampled density and error
					highResSample.setDensity(sampleDensityForMatchpoint(highResSample));
					highResSample.setVelocity(sampleVelocityForMatchpoint(highResSample));
					highResSample.setCurl(sampleCurlForMatchpoint(highResSample));

					// update error
					densityError = matchpoint.getMass() * (matchpoint.getDensity() - highResSample.getDensity());
					absError_density = abs(((matchpoint.getDensity() - highResSample.getDensity()) / matchpoint.getDensity()));

					velocityError = matchpoint.getMass() * (matchpoint.getVelocity() - highResSample.getVelocity());
					absError_velocity = length((matchpoint.getVelocity() - highResSample.getVelocity())) / length(matchpoint.getVelocity());

					curlError = matchpoint.getMass() * (matchpoint.getCurl() - highResSample.getCurl());
					absError_curl = length((matchpoint.getCurl() - highResSample.getCurl())) / length(matchpoint.getCurl());

					iterations++;
				}

				/*cout << "Raw Density error is " << densityError << endl;
				cout << "Final Density error is " << absError_density << endl;
				cout << "Final Velocity error is " << absError_velocity << endl;
				cout << "Final Curl error is " << absError_curl << endl;*/
			}

			nextKeyframe++;
		}
	}

	steps++;
}

// render the GUI
void renderGui(bool& isPaused, std::string& buttonText) {
	// Create GUI
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	// hides the GUI if space is pressed
	if (!keyToggles[(unsigned)' ']) {

		ImGui::SetNextWindowSize(ImVec2(840, 600));
		ImGui::Begin("Control Window");
		ImGui::SetWindowFontScale(2.0f);
		ImGui::Text("Press space to hide window\n");
		ImGui::TextWrapped("WARNING: While not stricly required, it is HIGHLY recommended to reset the the model after parameter changes\n");

		ImGui::Separator();
		ImGui::Text("Liquid Physics Parameters");
		if (ImGui::BeginCombo("", NULL)) {
			ImGui::SliderFloat("Viscosity", &VISCOSITY, 0.0f, 1.0f);
			ImGui::SliderFloat("Elasticity", &ELASTICITY, 0.0f, 1.0f);
			ImGui::InputFloat("At Rest Density", &DENSITY_0_GUESS);
			ImGui::SliderFloat("Friction", &FRICTION, 0.0f, 1.0f);
			ImGui::SliderFloat("Stiffness", &STIFFNESS_PARAM, 0.0f, 100.0f);
			ImGui::SliderFloat("\"Y\" Parameter", &Y_PARAM, 0.0f, 50.0f);
			ImGui::EndCombo();
		}

		ImGui::Separator();
		ImGui::Text("Surface Tension Parameters");
		if (ImGui::BeginCombo(" ", NULL)) {
			ImGui::SliderFloat("Tension Threshold", &TENSION_THRESHOLD, 0.0f, MAX_RADIUS);
			ImGui::SliderFloat("Alpha", &TENSION_ALPHA, 0.0f, 5.0f);
			ImGui::EndCombo();
		}

		ImGui::Separator();
		ImGui::Text("Simulation Parameters");
		if (ImGui::BeginCombo("  ", NULL)) {
			ImGui::SliderFloat("Time Per Step", &TIMESTEP, 0.0f, 0.05f);
			ImGui::SliderInt("Steps Per Kd-Tree Update", &steps_per_update, 1, 20);
			ImGui::EndCombo();
		}

		ImGui::Separator();
		ImGui::Text("Add Matchpoint");
		if (ImGui::BeginCombo("       ", NULL)) {
			ImGui::InputFloat3("Matchpoint Position", matchPointPosition);
			ImGui::SliderFloat("Matchpoint Radius", &matchPointRadius, 0.0f, 10.0f);
			if (ImGui::Button("Add Matchpoint")) {
				// initialize a particle to act as a matchpoint
				Particle matchPoint;
				glm::vec3 mPosition = glm::vec3(matchPointPosition[0], matchPointPosition[1], matchPointPosition[2]);
				matchPoint.setPosition(mPosition);
				matchPoint.setRadius(matchPointRadius);
				matchPoint.setMass(1.0f);
				Vec3f matchpointPosition = Vec3f(matchPoint.getPosition().x, matchPoint.getPosition().y, matchPoint.getPosition().z);
				particlePositions[particleCount + matchpointNumber] = matchpointPosition;
				matchPoint.setIsMatchpoint(true);

				// Add it to the particle list
				defaultMatchpoints.push_back(matchPoint);
				particleList[particleCount + matchpointNumber] = matchPoint;
				matchpointNumber++;
			}
			ImGui::EndCombo();
		}

		// should always be last
		if (ImGui::Button(buttonText.c_str())) {
			isPaused = !isPaused;
			if (buttonText.compare("Play") == 0) {
				buttonText = "Pause";
			}
			else {
				buttonText = "Play";
			}
		}
		if (ImGui::Button("Reset")) {
			buttonText = "Play";
			isPaused = true;
			timePassed = 0.0f;
			previousParticleCount = particleCount;
			particleCount = LOW_RES_COUNT;
			keyframes.clear();
			if (selected_scene == Scene::DAM_BREAK) {
				initSceneDamBreak();
			}
			else if (selected_scene == Scene::SPLASH) {
				initSceneSplash();
			}
			else {
				initSceneOriginal();
			}
			initMatchPoints();
			initKdTree();
			for (int i = 0; i < particleCount; i++) {
				setNeighbors(particleList[i], i);
			}
		}
		if (ImGui::Button("Record in High Resolution")) {
			isPaused = true;
			cudaDeviceSynchronize();
			recording = true;

			end_time = timePassed;
			timePassed = 0.0f;
			kdTree = nullptr;
			previousParticleCount = particleCount;
			particleCount = HIGH_RES_COUNT;
			particleForShape = HIGH_RES_COUNT_SHAPE;
			SMOOTHING_RADIUS = HIGH_RES_RADIUS;
			kernel->setSmoothingRadius(SMOOTHING_RADIUS);
			initAverageMass();
			MAX_RADIUS = HIGH_RES_RADIUS;
			float scaleFactor = (powf(resolutionConstant, (1.f / 3.f)) / powf(particleCount, (1.f / 3.f)));
			scaleParticles = glm::vec3(.5f * (scaleFactor), .5f * (scaleFactor), .5f * (scaleFactor));
			PARTICLES_PER_THREAD = HIGH_RES_COUNT / N_THREADS;
			TIMESTEP = .0125f;
			// DENSITY_0_GUESS = DENSITY_0_GUESS / scaleFactor;

			if (selected_scene == Scene::DAM_BREAK) {
				initSceneDamBreak();
			}
			else if (selected_scene == Scene::SPLASH) {
				initSceneSplash();
			}
			else {
				initSceneOriginal();
			}
			cout << "Updated particle count is " << particleCount << endl;
			initKdTree();
			for (int i = 0; i < particleCount; i++) {
				setNeighbors(particleList[i], i);
			}
			isPaused = false;
			cout << "Number of keyframes: " << keyframes.size() << endl;
			cout << "Recording... please be patient :)" << endl;
			// density_constant = DENSITY_0_GUESS / averageDensity;
			// DENSITY_0_GUESS = averageDensity;
			keyToggles[(unsigned)' '] = true;
		}
		if (ImGui::Button("Record in Mid Resolution")) {
			cudaDeviceSynchronize();
			isPaused = true;
			recording = true;

			end_time = timePassed;
			timePassed = 0.0f;
			// FIXME: leftover particles after switch
			kdTree = nullptr;
			previousParticleCount = particleCount;
			particleCount = MID_RES_COUNT;
			particleForShape = MID_RES_COUNT_SHAPE;
			SMOOTHING_RADIUS = MID_RES_RADIUS;
			kernel->setSmoothingRadius(SMOOTHING_RADIUS);
			initAverageMass();
			MAX_RADIUS = MID_RES_RADIUS;
			float scaleFactor = (powf(resolutionConstant, (1.f / 3.f)) / powf(particleCount, (1.f / 3.f)));
			scaleParticles = glm::vec3(.5f * (scaleFactor), .5f * (scaleFactor), .5f * (scaleFactor));
			PARTICLES_PER_THREAD = HIGH_RES_COUNT / N_THREADS;
			TIMESTEP = .0125f;
			// DENSITY_0_GUESS = DENSITY_0_GUESS / scaleFactor;

			if (selected_scene == Scene::DAM_BREAK) {
				initSceneDamBreak();
			}
			else if (selected_scene == Scene::SPLASH) {
				initSceneSplash();
			}
			else {
				initSceneOriginal();
			}
			cout << "Updated particle count is " << particleCount << endl;
			initKdTree();
			for (int i = 0; i < particleCount; i++) {
				setNeighbors(particleList[i], i);
			}
			isPaused = false;

			cout << "Recording... please be patient :)" << endl;
			// density_constant = DENSITY_0_GUESS / averageDensity;
			// DENSITY_0_GUESS = averageDensity;
			keyToggles[(unsigned)' '] = true;
		}
		if (ImGui::Button("Record in Low Resolution")) {
			recording_low_res = true;
			keyToggles[(unsigned)' '] = true;
		}
		ImGui::End();
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	}
}

int main(int argc, char** argv)
{
	// Set error callback.
	glfwSetErrorCallback(error_callback);
	// Initialize the library.
	if (!glfwInit()) {
		return -1;
	}
	// Create a windowed mode window and its OpenGL context.
	window = glfwCreateWindow(1280, 960, "SPH Simulator", NULL, NULL);
	if (!window) {
		glfwTerminate();
		return -1;
	}
	// Make the window's context current.
	glfwMakeContextCurrent(window);
	// Initialize GLEW.
	glewExperimental = true;
	if (glewInit() != GLEW_OK) {
		cerr << "Failed to initialize GLEW" << endl;
		return -1;
	}
	glGetError(); // A bug in glewInit() causes an error that we can safely ignore.
	cout << "OpenGL version: " << glGetString(GL_VERSION) << endl;
	cout << "GLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << endl;
	// Set vsync.
	glfwSwapInterval(1);
	// Set keyboard callback.
	glfwSetKeyCallback(window, key_callback);
	// Set char callback.
	glfwSetCharCallback(window, char_callback);
	// Set cursor position callback.
	glfwSetCursorPosCallback(window, cursor_position_callback);
	// Set mouse button callback.
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	// Initialize scene.
	init();

	// Loop until the user closes the window.

	// initialize guess density;

	// set neighbors for all particles
	for (int i = 0; i < particleCount; i++) {
		setNeighbors(particleList[i], i);
	}

	float timeStart = glfwGetTime();

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 460");

	std::string buttonText = "Play";
	bool isPaused = true;
	cout << "Number of surfaces: " << numSurfaces << endl;
	while (!glfwWindowShouldClose(window) || (recording && timePassed <= end_time)) {
		if (!glfwGetWindowAttrib(window, GLFW_ICONIFIED)) {

			// Simulate and draw water
			if (!isPaused) {
				float timeEnd = glfwGetTime();
				// Integrate partices
				updateFluid(TIMESTEP);
				// Render scene.
				timePassed += (TIMESTEP);
				totalTime += timePassed;
				if (timePassed >= 6 && numSurfaces == 6 && selected_scene == Scene::DAM_BREAK) {
					cout << "Release the kracken!" << endl;
					numSurfaces = 5;
				}
				GLSL::checkError(GET_FILE_LINE);
			}

			render();
			renderGui(isPaused, buttonText);
			if (recording && timePassed > end_time) {
				cout << "Recording finished" << endl;
				break;
			}

			// Swap front and back buffers.
			glfwSwapBuffers(window);

			if (recording || recording_low_res) {
				glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, buffer);
				fwrite(buffer, sizeof(int) * width * height, 1, ffmpeg);
			}
		}
		// Poll for and process events.
		glfwPollEvents();
	}

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	// close video stream
	_pclose(ffmpeg);

	// clean up memory
	cudaFree(particleList);
	cudaFree(kernel);
	cudaFree(surfaces);

	// Quit program.
	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}
