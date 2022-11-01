#include <iostream>
#include <vector>
#include <random>

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

#include "GLSL.h"
#include "Program.h"
#include "MatrixStack.h"
#include "Particle.h"
#include "cyPointCloud.h"
#include "Shape.h"
#include "cyVector.h"
#include "Plane.h"
#include "imgui-master/imgui.h"
#include "imgui-master/backends/imgui_impl_glfw.h"
#include "imgui-master/backends/imgui_impl_opengl3.h"
#include "Kernel.h"
#include <thread>

using namespace std;
using cy::Vec3f;

enum class Scene {
	DEFAULT,
	DAM_BREAK
};

GLFWwindow *window; // Main application window
string RESOURCE_DIR = "..\\resources\\"; // Where the resources are loaded from
shared_ptr<Program> prog;

bool keyToggles[256] = {false}; // only for English keyboards!
glm::vec2 cameraRotations(0, 0);
glm::vec2 mousePrev(-1, -1);


float DENSITY_0_GUESS = 1.0f; // density of water= 1 g/cm^3
float STIFFNESS_PARAM = 7.0f;
float Y_PARAM = 7.0f;
uint32_t LOW_RES_COUNT = 10000;
uint32_t HIGH_RES_COUNT = 100000;
float LOW_RES_RADIUS = 3.0f;
float MAX_RADIUS = LOW_RES_RADIUS;
float SMOOTHING_RADIUS = LOW_RES_RADIUS;
float VISCOSITY = 1.0f;
float TIMESTEP = .01f;

float FRICTION = .1f;
float ELASTICITY = .7f;

// surface tension stuff
float TENSION_ALPHA = 0.3f;
float TENSION_THRESHOLD = .5f;
float DELTA_ERROR = 0;

float totalTime = 0.0f;

int particleCount = LOW_RES_COUNT;
int steps = 0;
int steps_per_update = 3;
Particle* particleList;
Vec3f* particlePositions;

shared_ptr<cy::PointCloud<Vec3f, float, 3>> kdTree;
shared_ptr<Shape> lowResSphere;
std::vector<Plane> surfaces;

glm::vec3 scaleStructure = glm::vec3(.05f, .05f, .05f);
glm::vec3 scaleParticles = glm::vec3(.5f, .5f, .5f);

Scene selected_scene = Scene::DAM_BREAK;

const int N_THREADS = 10;
int PARTICLES_PER_THREAD = particleCount / N_THREADS;
std::thread threads[N_THREADS];

const float MAX_DENSITY_ERROR = .01f;
const int MIN_ITERATIONS = 3;

static void error_callback(int error, const char *description)
{
	cerr << description << endl;
}

static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
	if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
}

static void char_callback(GLFWwindow *window, unsigned int key)
{
	keyToggles[key] = !keyToggles[key];
}

static void cursor_position_callback(GLFWwindow* window, double xmouse, double ymouse)
{
	if(mousePrev.x >= 0) {
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
	if(action == GLFW_PRESS) {
		mousePrev.x = xmouse;
		mousePrev.y = ymouse;
	} else {
		mousePrev[0] = -1;
		mousePrev[1] = -1;
	}
}

float calculateDensityForParticle(const Particle& x, bool predicted = false) {
	float density = x.getMass() * Kernel::polyKernelFunction(x, x, predicted);
	for (int j = 0; j < x.getNeighbors().size(); j++) {
		Particle* xj = x.getNeighbors().at(j);
		density += (xj->getMass() * Kernel::polyKernelFunction(x, *xj, predicted));
	}

	return density;
}

float calculatePressureForParticle(const Particle& x) {
	float pressure = ((STIFFNESS_PARAM * DENSITY_0_GUESS) / Y_PARAM) * (powf((x.getDensity() / DENSITY_0_GUESS), Y_PARAM) - 1.0f);
	return pressure;
}

glm::vec3 pressureGradient(const Particle& xi, bool predicted = false) {
	glm::vec3 pressureGradient = glm::vec3(0, 0, 0);

	// for every Particle xj in the neighbor hood of xi
	for (int j = 0; j < xi.getNeighbors().size(); j++) {
		Particle* xj = xi.getNeighbors().at(j);
		glm::vec3 r = xi.getPosition() - xj->getPosition();

		float pressureTerm = (xi.getPressure() / powf(xi.getDensity(), 2.0f)) + (xj->getPressure() / powf(xj->getDensity(), 2.0f));
		pressureGradient += (xj->getMass() * pressureTerm * Kernel::spikyKernelGradient(xi, *xj, predicted));
	}
	return -1.0f * pressureGradient;
}

void setNeighbors(Particle& x, int xIndex) {
	cy::PointCloud<Vec3f, float, 3>::PointInfo* info = new cy::PointCloud<Vec3f, float, 3>::PointInfo[500];
	int numPointsInRadius = kdTree->GetPoints(particlePositions[xIndex], LOW_RES_RADIUS, 500, info);

	// create a vector for the new neighbors
	std::vector<Particle*> neighbors;
	for (int i = 0; i < numPointsInRadius; i++) {
		if (xIndex != info[i].index) {
			neighbors.push_back(&particleList[info[i].index]);
		}
	}
	x.setNeighbors(neighbors);
	delete[] info;
}

glm::vec3 diffusionTerm(const Particle& xi, bool predicted = false) {
	glm::vec3 diffusionLaplacian = glm::vec3(0.0f, 0.0f, 0.0f);

	// for every Particle xj in the neighbor hood of xi
	for (int j = 0; j < xi.getNeighbors().size(); j++) {
		Particle* xj = xi.getNeighbors().at(j);
		glm::vec3 velocityTerm = (xj->getVelocity() - xi.getVelocity()) / xj->getDensity();

		diffusionLaplacian += (xj->getMass() * velocityTerm * Kernel::viscosityKernelLaplacian(xi, *xj, predicted));
	}
	return diffusionLaplacian;
}

// surface tension functions
glm::vec3 surfaceNormalField(const Particle& xi) {
	glm::vec3 surfaceField = glm::vec3(0.0f, 0.0f, 0.0f);

	// for every Particle xj in the neighbor hood of xi
	for (int j = 0; j < xi.getNeighbors().size(); j++) {

		Particle* xj = xi.getNeighbors().at(j);
		if (xj != &xi) {
			float outside_term = xj->getMass() * 1 / xj->getDensity();
			surfaceField += (outside_term * Kernel::polyKernelGradient(xi, *xj));
		}

	}
	return surfaceField;
}

// surface tension functions
glm::vec3 colorFieldLaplacian(const Particle& xi) {
	glm::vec3 surfaceField = glm::vec3(0.0f, 0.0f, 0.0f);

	// for every Particle xj in the neighbor hood of xi
	for (int j = 0; j < xi.getNeighbors().size(); j++) {
		Particle* xj = xi.getNeighbors().at(j);
		float outside_term = xj->getMass() * 1 / xj->getDensity();

		surfaceField += (outside_term * Kernel::polyKernelLaplacian(xi, *xj));
	}
	return surfaceField;
}

void setNeighborsForParticles(int start_index, int end_index) {
	for (int i = start_index; i < end_index; i++) {
		setNeighbors(particleList[i], i);
	}
}

void setDensitiesForParticles(int start_index, int end_index, bool predicted = false) {
	for (int i = start_index; i < end_index; i++) {
		particleList[i].setDensity(calculateDensityForParticle(particleList[i], predicted));
	}
}

void setSurfaceTensionForParticles(int start_index, int end_index) {
	for (int i = start_index; i < end_index; i++) {
		particleList[i].setSurfaceNormal(surfaceNormalField(particleList[i]));
	}
}

void setColorFieldLaplaciansForParticles(int start_index, int end_index) {
	for (int i = start_index; i < end_index; i++) {
		particleList[i].setColorFieldLaplacian(colorFieldLaplacian(particleList[i]));
	}
}

void setPressuresForParticles(int start_index, int end_index) {
	for (int i = start_index; i < end_index; i++) {
		particleList[i].setPressure(calculatePressureForParticle(particleList[i]));
	}
}

void initializeForcesForParticles(int start_index, int end_index) {
	for (int i = start_index; i < end_index; i++) {
		particleList[i].forces.pressure = glm::vec3(0, 0, 0);
		particleList[i].forces.viscosity = diffusionTerm(particleList[i]);
		particleList[i].forces.external = glm::vec3(0.0, -9.8f, 0.0f);
		particleList[i].setPressure(0);
	}
}

void setAccelerationForParticles_updated(int start_index, int end_index) {
	for (int i = start_index; i < end_index; i++) {
		glm::vec3 pressureForce = particleList[i].forces.pressure;
		glm::vec3 diffusionForce = particleList[i].forces.viscosity;
		glm::vec3 externalForce = particleList[i].forces.external;

		// calculate surface pressure/tension
		/*glm::vec3 k = -1.0f * particleList[i].getColorFieldLaplacian() / length(particleList[i].getSurfaceNormal());
		glm::vec3 tension = k * TENSION_ALPHA * particleList[i].getSurfaceNormal();*/

		glm::vec3 acceleration = pressureForce + VISCOSITY * diffusionForce + externalForce;
		/*if (length(tension) > TENSION_THRESHOLD) {
			acceleration += tension;
		}*/
		particleList[i].setAcceleration(acceleration);
	}
}

void setAccelerationForParticles(int start_index, int end_index) {
	for (int i = start_index; i < end_index; i++) {
		glm::vec3 pressureForce = pressureGradient(particleList[i]);
		glm::vec3 diffusionForce = diffusionTerm(particleList[i]);
		glm::vec3 externalForce = glm::vec3(0.0, -9.8f, 0.0f);

		// calculate surface pressure/tension
		glm::vec3 k = -1.0f * particleList[i].getColorFieldLaplacian() / length(particleList[i].getSurfaceNormal());
		glm::vec3 tension = k * TENSION_ALPHA * particleList[i].getSurfaceNormal();

		glm::vec3 acceleration = pressureForce + VISCOSITY * diffusionForce + externalForce;
		if (length(tension) > TENSION_THRESHOLD) {
			acceleration += tension;
		}
		particleList[i].setAcceleration(acceleration);
	}
}

void updatePredictedPositionForParticle(Particle& xi, double time) {
	float timeStepRemaining = time;

	glm::vec3 newVelocity = xi.getVelocity() + xi.getAcceleration() * timeStepRemaining;
	glm::vec3 averageVelocity = (newVelocity + xi.getVelocity()) / 2.0f;
	glm::vec3 newPosition = xi.getPosition() + newVelocity * timeStepRemaining;

	for (Plane surface : surfaces) {
		if (Particle::willCollideWithPlane(xi.getPosition(), newPosition, xi.getRadius(), surface)) {
			// collision stuff
			glm::vec3 velocityNormalBefore = glm::dot(newVelocity, surface.getNormal()) * surface.getNormal();
			glm::vec3 velocityTangentBefore = newVelocity - velocityNormalBefore;
			glm::vec3 velocityNormalAfter = -1 * ELASTICITY * velocityNormalBefore;
			float frictionMultiplier = min((1 - FRICTION) * glm::length(velocityNormalBefore), glm::length(velocityTangentBefore));
			glm::vec3 velocityTangentAfter;
			if (glm::length(velocityTangentBefore) == 0) {
				velocityTangentAfter = velocityTangentBefore;
			}
			else {
				velocityTangentAfter = velocityTangentBefore - frictionMultiplier * glm::normalize(velocityTangentBefore);
			}

			newVelocity = velocityNormalAfter + velocityTangentAfter;
			float distance = xi.getDistanceFromPlane(newPosition, xi.getRadius(), surface);
			glm::vec3 addedVector = glm::vec3(surface.getNormal()) * (distance * (1 + ELASTICITY));
			newPosition = newPosition + addedVector;
			// particleList[i].setPosition(newPosition);
		}
	}

	xi.setPredictedVelocity(newVelocity);
	xi.setPredictedPosition(newPosition);
}

void updatePredictedPositionsForParticles(int start_index, int end_index, double time) {
	for (int i = start_index; i < end_index; i++) {
		updatePredictedPositionForParticle(particleList[i], time);
	}
}

void updatePositionForParticles(int start_index, int end_index, double time) {
	for (int i = start_index; i < end_index; i++) {
		float timeStepRemaining = time;

		glm::vec3 newVelocity = particleList[i].getVelocity() + particleList[i].getAcceleration() * timeStepRemaining;
		glm::vec3 averageVelocity = (newVelocity + particleList[i].getVelocity()) / 2.0f;
		glm::vec3 newPosition = particleList[i].getPosition() + newVelocity * timeStepRemaining;

		for (Plane surface : surfaces) {
			if (Particle::willCollideWithPlane(particleList[i].getPosition(), newPosition, particleList[i].getRadius(), surface)) {
				// collision stuff
				glm::vec3 velocityNormalBefore = glm::dot(newVelocity, surface.getNormal()) * surface.getNormal();
				glm::vec3 velocityTangentBefore = newVelocity - velocityNormalBefore;
				glm::vec3 velocityNormalAfter = -1 * ELASTICITY * velocityNormalBefore;
				float frictionMultiplier = min((1 - FRICTION) * glm::length(velocityNormalBefore), glm::length(velocityTangentBefore));
				glm::vec3 velocityTangentAfter;
				if (glm::length(velocityTangentBefore) == 0) {
					velocityTangentAfter = velocityTangentBefore;
				}
				else {
					velocityTangentAfter = velocityTangentBefore - frictionMultiplier * glm::normalize(velocityTangentBefore);
				}

				newVelocity = velocityNormalAfter + velocityTangentAfter;
				float distance = particleList[i].getDistanceFromPlane(newPosition, particleList[i].getRadius(), surface);
				glm::vec3 addedVector = glm::vec3(surface.getNormal()) * (distance * (1 + ELASTICITY));
				newPosition = newPosition + addedVector;
				// particleList[i].setPosition(newPosition);
			}
		}

		particleList[i].setVelocity(newVelocity);
		particleList[i].setPosition(newPosition);
		particlePositions[i] = Vec3f(newPosition.x, newPosition.y, newPosition.z);

	}
}

void initParticleList_atRest() {
	particleList = new Particle[particleCount];
	particlePositions = new Vec3f[particleCount];

	// put them in a cube shape for ease of access
	float depth = 20.0f;
	int slice = particleCount / depth;
	int width = slice / 20.0f;
	int height = slice / width;

	float volume = (height * width * depth * .25);
	float volumePerParticle = volume / particleCount;
	float mass = volumePerParticle * DENSITY_0_GUESS;
	for (int i = 0; i < depth; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < height; k++) {
				Particle p;
				float x_position = (float)j;
				float y_position = (float)k * .25;
				float z_position = (float)i;
				if (k % 2 == 1) {
					x_position += .5;
				}
				p.setPosition(glm::vec3(x_position, y_position, z_position));
				p.setDensity(DENSITY_0_GUESS);
				p.setMass(mass);
				p.setVelocity(glm::vec3(0.0f, 0.0f, 0.0f));
				p.setRadius(scaleParticles.x);

				int index = (slice * i) + (height * j) + k;
				particleList[index] = p;
				particlePositions[index] = Vec3f(x_position, y_position, z_position);

			}
		}
	}
}

void initParticleList() {
	particleList = new Particle[particleCount];
	particlePositions = new Vec3f[particleCount];

	// put them in a cube shape for ease of access
	float depth = 20.0f;
	int slice = particleCount / depth;
	int width = slice / 20.0f;
	int height = slice / width;

	float volume = (height * width * depth);
	float volumePerParticle = volume / particleCount;
	float mass = volumePerParticle * DENSITY_0_GUESS;
	for (int i = 0; i < depth; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < height; k++) {
				Particle p;
				p.setPosition(glm::vec3(j, k, i));
				p.setDensity(DENSITY_0_GUESS);
				p.setMass(mass);
				p.setVelocity(glm::vec3(0.0f, 0.0f, 0.0f));
				p.setRadius(scaleParticles.x);

				int index = (slice * i) + (height * j) + k;
				particleList[index] = p;
				particlePositions[index] = Vec3f(j, k, i);

			}
		}
	}
}

void initSceneOriginal() {
	initParticleList();

	Plane ground(glm::vec3(0.0, 1.0, 0.0), glm::vec3(0.0, -1.0, 0.0));
	Plane wall_1(glm::vec3(0.0f, 0.0, -1.0), glm::vec3(0.0, 0.0, 26.0f));
	Plane wall_2(glm::vec3(1.0, 0.0, .0), glm::vec3(-1.0, 0.0, 0.0));
	Plane wall_3(glm::vec3(0.0, 0.0, 1.0), glm::vec3(0.0, 0.0, -1.0));
	Plane wall_4(glm::vec3(-1.0, 0.0, 0.0), glm::vec3(26.0f, 0.0, 0.0));

	// initialize surfaces
	surfaces.clear();
	surfaces.push_back(ground);
	surfaces.push_back(wall_1);
	surfaces.push_back(wall_2);
	surfaces.push_back(wall_3);
	surfaces.push_back(wall_4);

}

void initSceneDamBreak() {
	initParticleList_atRest();

	Plane ground(glm::vec3(0.0, 1.0, 0.0), glm::vec3(0.0, 0.0f - scaleParticles.x , 0.0));
	Plane wall_1(glm::vec3(0.0f, 0.0, -1.0), glm::vec3(0.0, 0.0, 20.0f + scaleParticles.x));
	Plane wall_2(glm::vec3(1.0, 0.0, .0), glm::vec3(0.0 - scaleParticles.x, 0.0f, 0.0));
	Plane wall_3(glm::vec3(0.0, 0.0, 1.0), glm::vec3(0.0, 0.0, 0.0 - scaleParticles.x));
	Plane wall_4(glm::vec3(-1.0, 0.0, 0.0), glm::vec3(25 + scaleParticles.x, 0.0, 0.0));

	// initialize surfaces
	surfaces.clear();
	surfaces.push_back(ground);
	surfaces.push_back(wall_1);
	surfaces.push_back(wall_2);
	surfaces.push_back(wall_3);
	surfaces.push_back(wall_4);

}

void initKdTree() {
	// Should automatically build the tree?
	kdTree = make_shared<cy::PointCloud<Vec3f, float, 3>>(particleCount, particlePositions);
}

void initDeltaError() {
	initParticleList_atRest();
	initKdTree();

	// set neighbors
	for (int i = 0; i < N_THREADS; i++) {
		threads[i] = thread(setNeighborsForParticles, i * PARTICLES_PER_THREAD, (i + 1) * PARTICLES_PER_THREAD);
	}

	for (int i = 0; i < N_THREADS; i++) {
		threads[i].join();
	}

	// find the particle with the most filled neihborhood
	Particle* mostFilled = nullptr;
	for (int i = 0; i < particleCount; i++) {
		if (!mostFilled || particleList[i].getNeighbors().size() > mostFilled->getNeighbors().size()) {
			mostFilled = &particleList[i];
		}
	}

	// TODO: calculate sigma
	// TODO: make faster by not calculating things about particles we don't care about smh
	// initialize the forces and pressures
	for (int i = 0; i < N_THREADS; i++) {
		threads[i] = thread(initializeForcesForParticles, i * PARTICLES_PER_THREAD, (i + 1) * PARTICLES_PER_THREAD);
	}

	for (int i = 0; i < N_THREADS; i++) {
		threads[i].join();
	}

	// initialize the accelerations
	for (int i = 0; i < N_THREADS; i++) {
		threads[i] = thread(setAccelerationForParticles_updated, i * PARTICLES_PER_THREAD, (i + 1) * PARTICLES_PER_THREAD);
	}

	for (int i = 0; i < N_THREADS; i++) {
		threads[i].join();
	}

	// update position for given particle
	for (int i = 0; i < N_THREADS; i++) {
		threads[i] = thread(updatePredictedPositionsForParticles, i * PARTICLES_PER_THREAD, (i + 1) * PARTICLES_PER_THREAD, TIMESTEP);
	}

	for (int i = 0; i < N_THREADS; i++) {
		threads[i].join();
	}

	glm::vec3 densityKernelGradientSum = glm::vec3(0, 0, 0);
	glm::vec3 pressureKernelGradientSum = glm::vec3(0, 0, 0);
	float dotSum = 0.0f;
	cout << "Number of neighbors " << mostFilled->getNeighbors().size() << endl;
	for (int i = 0; i < mostFilled->getNeighbors().size(); i++) {
		Particle* xj = mostFilled->getNeighbors()[i];
		glm::vec3 densityGradient = Kernel::polyKernelGradient(*mostFilled, *xj, true);
		glm::vec3 pressureGradient = Kernel::spikyKernelGradient(*mostFilled, *xj, true);
		densityKernelGradientSum += densityGradient;
		pressureKernelGradientSum += pressureGradient;
		dotSum += glm::dot(densityGradient, pressureGradient);
	}

	float beta = powf(TIMESTEP, 2.0) * powf(mostFilled->getMass(), 2.0f) * 2.0f / powf(DENSITY_0_GUESS, 2.0f);
	float denominator = beta * (glm::dot(-1.0f * pressureKernelGradientSum, densityKernelGradientSum) - dotSum);
	DELTA_ERROR = -1.0f / denominator;
	cout << "DELTA ERROR IS " << DELTA_ERROR << endl << endl;

	// also make this multithreaded
	// reset all densities
	for (int i = 0; i < particleCount; i++) {
		particleList[i].setDensity(DENSITY_0_GUESS);
	}
}

static void init()
{
	GLSL::checkVersion();

	Kernel::setSmoothingRadius(SMOOTHING_RADIUS);
	
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

	// Initialize sigma
	initDeltaError();
	
	keyToggles[(unsigned)'l'] = true;

	// initialize particles and tree
	switch (selected_scene) {
		case Scene::DAM_BREAK:
			initSceneDamBreak();
			break;
		default:
			initSceneOriginal();
	}
	
	initKdTree();
	// update neighbors
	for (int i = 0; i < N_THREADS; i++) {
		threads[i] = thread(setNeighborsForParticles, i * PARTICLES_PER_THREAD, (i + 1) * PARTICLES_PER_THREAD);
	}

	for (int i = 0; i < N_THREADS; i++) {
		threads[i].join();
	}
	

	// initialize shape for sphere
	lowResSphere = make_shared<Shape>();
	lowResSphere->loadMesh(RESOURCE_DIR + "low_res_sphere.obj");
	lowResSphere->init();

	// If there were any OpenGL errors, this will print something.
	// You can intersperse this line in your code to find the exact location
	// of your OpenGL error.
	GLSL::checkError(GET_FILE_LINE);
}

void drawParticles(shared_ptr<MatrixStack>& MV) {
	MV->pushMatrix();
	glUniform3f(prog->getUniform("kd"), 0.0f, 0.3f, .7f);
	glUniform3f(prog->getUniform("ka"), 0.0f, 0.3f, .7f);
	glUniform3f(prog->getUniform("ks"), 0.0f, 0.3f, .7f);
	glUniform3f(prog->getUniform("lightPos"), 10.0f, 10.0f, -3.0f);
	MV->scale(scaleStructure);
	for (int i = 0; i < particleCount; i++) {
		MV->pushMatrix();
		MV->translate(particleList[i].getPosition());
		MV->scale(scaleParticles);
		glUniformMatrix4fv(prog->getUniform("MV"), 1, GL_FALSE, glm::value_ptr(MV->topMatrix()));
		lowResSphere->draw(prog);
		MV->popMatrix();
	}
	MV->popMatrix();
}

void render()
{
	// Update time.
	double t = glfwGetTime();
	// Get current frame buffer size.
	int width, height;
	glfwGetFramebufferSize(window, &width, &height);
	glViewport(0, 0, width, height);
	
	// Clear buffers
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	auto P = make_shared<MatrixStack>();
	auto MV = make_shared<MatrixStack>();
	P->pushMatrix();
	MV->pushMatrix();
	
	double aspect = (double)width/height;
	P->multMatrix(glm::ortho(-aspect, aspect, -1.0, 1.0, -2.0, 2.0));
	MV->rotate(cameraRotations.y, glm::vec3(1, 0, 0));
	MV->rotate(cameraRotations.x, glm::vec3(0, 1, 0));
	
	// Bind the program
	prog->bind();
	glUniformMatrix4fv(prog->getUniform("P"), 1, GL_FALSE, glm::value_ptr(P->topMatrix()));
	glUniformMatrix4fv(prog->getUniform("MV"), 1, GL_FALSE, glm::value_ptr(MV->topMatrix()));

	// draw particles
	drawParticles(MV);

	// Unbind the program
	prog->unbind();

	// Pop matrix stacks.
	MV->popMatrix();
	P->popMatrix();

	GLSL::checkError(GET_FILE_LINE);
}

void updateFluid(float time) {

	// update the kd tree
	//std::thread kdTree_thread;
	//kdTree_thread = thread(initKdTree);

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

	 //update density
	for (int i = 0; i < N_THREADS; i++) {
		// particleList[i].setDensity(calculateDensityForParticle(particleList[i]));
		threads[i] = thread(setDensitiesForParticles, i * PARTICLES_PER_THREAD, (i + 1) * PARTICLES_PER_THREAD, false);
	} 

	for (int i = 0; i < N_THREADS; i++) {
		threads[i].join();
	}

	// TODO: optimize with one loop later
	// update surface normals
	for (int i = 0; i < N_THREADS; i++) {
		// particleList[i].setSurfaceNormal(surfaceNormalField(particleList[i]));
		threads[i] = thread(setSurfaceTensionForParticles, i * PARTICLES_PER_THREAD, (i + 1) * PARTICLES_PER_THREAD);
	}

	for (int i = 0; i < N_THREADS; i++) {
		threads[i].join();
	}

	for (int i = 0; i < N_THREADS; i++) {
		// particleList[i].setColorFieldLaplacian(colorFieldLaplacian(particleList[i]));
		threads[i] = thread(setColorFieldLaplaciansForParticles, i * PARTICLES_PER_THREAD, (i + 1) * PARTICLES_PER_THREAD);
	}

	for (int i = 0; i < N_THREADS; i++) {
		threads[i].join();
	}

	// update the pressures
	for (int i = 0; i < N_THREADS; i++) {
		// particleList[i].setPressure(calculatePressureForParticle(particleList[i]));
		threads[i] = thread(setPressuresForParticles, i * PARTICLES_PER_THREAD, (i + 1) * PARTICLES_PER_THREAD);
	}

	for (int i = 0; i < N_THREADS; i++) {
		threads[i].join();
	}

	// update the pressures
	for (int i = 0; i < N_THREADS; i++) {
		// particleList[i].setPressure(calculatePressureForParticle(particleList[i]));
		threads[i] = thread(setAccelerationForParticles, i * PARTICLES_PER_THREAD, (i + 1) * PARTICLES_PER_THREAD);
	}

	for (int i = 0; i < N_THREADS; i++) {
		threads[i].join();
	}

	//kdTree_thread.join();

	for (int i = 0; i < N_THREADS; i++) {
		// particleList[i].setPressure(calculatePressureForParticle(particleList[i]));
		threads[i] = thread(updatePositionForParticles, i * PARTICLES_PER_THREAD, (i + 1) * PARTICLES_PER_THREAD, time);
	}

	for (int i = 0; i < N_THREADS; i++) {
		threads[i].join();
	}

	steps++;
}

void updateFluidPCISPH(float time) {

	// find neighborhoods
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

	// initialize the forces and pressures
	for (int i = 0; i < N_THREADS; i++) {
		threads[i] = thread(initializeForcesForParticles, i * PARTICLES_PER_THREAD, (i + 1) * PARTICLES_PER_THREAD);
	}

	for (int i = 0; i < N_THREADS; i++) {
		threads[i].join();
	}

	// initialize the accelerations
	for (int i = 0; i < N_THREADS; i++) {
		threads[i] = thread(setAccelerationForParticles_updated, i * PARTICLES_PER_THREAD, (i + 1) * PARTICLES_PER_THREAD);
	}

	for (int i = 0; i < N_THREADS; i++) {
		threads[i].join();
	}
	
	float max_density_error = 0;
	int iter = 0;
	while (max_density_error > MAX_DENSITY_ERROR || iter < MIN_ITERATIONS) {

		// update predicted velocities and positions
		max_density_error = 0;
		for (int i = 0; i < N_THREADS; i++) {
			threads[i] = thread(updatePredictedPositionsForParticles, i * PARTICLES_PER_THREAD, (i + 1) * PARTICLES_PER_THREAD, time);
		}

		for (int i = 0; i < N_THREADS; i++) {
			threads[i].join();
		}

		// update predicted densities and pressures
		for (int i = 0; i < N_THREADS; i++) {
			threads[i] = thread(setDensitiesForParticles, i * PARTICLES_PER_THREAD, (i + 1) * PARTICLES_PER_THREAD, true);
		}

		for (int i = 0; i < N_THREADS; i++) {
			threads[i].join();
		}

		// TODO: make this step threaded
		int max_index = 0;
		for (int i = 0; i < particleCount; i++) {
			float density_error = particleList[i].getDensity() - DENSITY_0_GUESS;

			// set the error, if needed
			if (abs(density_error / DENSITY_0_GUESS) > max_density_error) {
				max_index = i;
				max_density_error = abs(density_error / DENSITY_0_GUESS);
			}

			// update pressure
			particleList[i].setPressure(particleList[i].getPressure() + (DELTA_ERROR * density_error));
		}

		// TODO: make this step threaded
		for (int i = 0; i < particleCount; i++) {
			// QUESTION: is kernel with predictions or no?
			particleList[i].forces.pressure = pressureGradient(particleList[i], false);
		}

		// recalculate accelerations
		for (int i = 0; i < N_THREADS; i++) {
			threads[i] = thread(setAccelerationForParticles_updated, i * PARTICLES_PER_THREAD, (i + 1) * PARTICLES_PER_THREAD);
		}

		for (int i = 0; i < N_THREADS; i++) {
			threads[i].join();
		}

		iter++;
		cout << "Iteration " << iter << endl;
		cout << "Error: " << max_density_error << endl;
		cout << "Max index " << max_index << endl;
		cout << "Density @ max index " << particleList[max_index].getDensity() << endl;
		cout << "Pressure @ max index " << particleList[max_index].getPressure() << endl;
		cout << "Predicted Position @ max index " << particleList[max_index].getPredictedPosition().x << ", " << particleList[max_index].getPredictedPosition().y << ", " << particleList[max_index].getPredictedPosition().z << endl;
		cout << "Pressure force @ max index " << particleList[max_index].forces.pressure.x << ", "<< particleList[max_index].forces.pressure.y << ", " << particleList[max_index].forces.pressure.z << endl << endl;
	}

	// update actual positions and velocities
	for (int i = 0; i < N_THREADS; i++) {
		// particleList[i].setPressure(calculatePressureForParticle(particleList[i]));
		threads[i] = thread(updatePositionForParticles, i * PARTICLES_PER_THREAD, (i + 1) * PARTICLES_PER_THREAD, time);
	}

	for (int i = 0; i < N_THREADS; i++) {
		threads[i].join();
	}
	steps++;
	cout << "Finished step " << steps << endl;
}

void renderGui(bool& isPaused, std::string& buttonText) {
	// Create GUI
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	// hides the GUI if space is pressed
	if (!keyToggles[(unsigned)' ']) {

		ImGui::SetNextWindowSize(ImVec2(840, 480));
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
			ImGui::SliderFloat("Stiffness", &STIFFNESS_PARAM, 0.0f, 10.0f);
			ImGui::SliderFloat("\"Y\" Parameter", &Y_PARAM, 0.0f, 10.0f);
			ImGui::EndCombo();
		}

		ImGui::Separator();
		ImGui::Text("Surface Tension Parameters");
		if (ImGui::BeginCombo(" ", NULL)) {
			ImGui::SliderFloat("Tension Threshold", &TENSION_THRESHOLD, 0.0f, MAX_RADIUS);
			ImGui::SliderFloat("Alpha", &TENSION_ALPHA, 0.0f, 1.0f);
			ImGui::EndCombo();
		}

		ImGui::Separator();
		ImGui::Text("Simulation Parameters");
		if (ImGui::BeginCombo("  ", NULL)) {
			ImGui::SliderFloat("Time Per Step", &TIMESTEP, 0.0f, 0.05f);
			ImGui::SliderInt("Steps Per Kd-Tree Update", &steps_per_update, 1, 20);
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
			if (selected_scene == Scene::DAM_BREAK) {
				initSceneDamBreak();
			}
			else {
				initSceneOriginal();
			}
			initKdTree();
			for (int i = 0; i < particleCount; i++) {
				setNeighbors(particleList[i], i);
			}

			float averageDensity = 0;
			for (int i = 0; i < particleCount; i++) {
				averageDensity += calculateDensityForParticle(particleList[i]) / (float)particleCount;
			}
			// density_constant = DENSITY_0_GUESS / averageDensity;
			// DENSITY_0_GUESS = averageDensity;
		}
		ImGui::End();
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	}
}

int main(int argc, char **argv)
{	
	// Set error callback.
	glfwSetErrorCallback(error_callback);
	// Initialize the library.
	if(!glfwInit()) {
		return -1;
	}
	// Create a windowed mode window and its OpenGL context.
	window = glfwCreateWindow(1280, 960, "SPH Simulator", NULL, NULL);
	if(!window) {
		glfwTerminate();
		return -1;
	}
	// Make the window's context current.
	glfwMakeContextCurrent(window);
	// Initialize GLEW.
	glewExperimental = true;
	if(glewInit() != GLEW_OK) {
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
	float timePassed = 0.0f;

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 460");

	std::string buttonText = "Play";
	bool isPaused = true;
	cout << "Number of surfaces: " << surfaces.size() << endl;
	while(!glfwWindowShouldClose(window)) {
		if(!glfwGetWindowAttrib(window, GLFW_ICONIFIED)) {
			
			// Simulate and draw water
			if (!isPaused) {
				float timeEnd = glfwGetTime();
				// Integrate partices
				updateFluidPCISPH(TIMESTEP);
				// Render scene.
				timePassed += (TIMESTEP);
				totalTime += timePassed;
				if (timePassed >= 4 && surfaces.size() == 5 && selected_scene == Scene::DAM_BREAK) {
					cout << "Release the kracken!" << endl;
					surfaces.resize(4);
				}
				GLSL::checkError(GET_FILE_LINE);
			}
			
			render();

			renderGui(isPaused, buttonText);
			
			// Swap front and back buffers.
			glfwSwapBuffers(window);
		}
		// Poll for and process events.
		glfwPollEvents();
	}

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	// clean up memory
	delete[] particleList;

	// Quit program.
	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}
