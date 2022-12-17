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

using namespace std;
using cy::Vec3f;

enum class Scene {
	DEFAULT,
	DAM_BREAK,
	SPLASH
};

struct Keyframe {
	std::vector<Particle> matchpoints;
	float time;
};

GLFWwindow *window; // Main application window
string RESOURCE_DIR = "..\\resources\\"; // Where the resources are loaded from
shared_ptr<Program> prog;

bool keyToggles[256] = {false}; // only for English keyboards!
glm::vec2 cameraRotations(0, 0);
glm::vec2 mousePrev(-1, -1);



float resolutionConstant = 8000;
float DENSITY_0_GUESS = .1f; // density of water= 1 g/cm^3
float STIFFNESS_PARAM = 60.0f;
float Y_PARAM = 7.0f;
uint32_t LOW_RES_COUNT = 8000;
uint32_t MID_RES_COUNT = 27000;
uint32_t HIGH_RES_COUNT = 64000;
uint32_t LOW_RES_COUNT_SHAPE = 250;
uint32_t MID_RES_COUNT_SHAPE = 1000;
uint32_t HIGH_RES_COUNT_SHAPE = 2000;
int particleCount = LOW_RES_COUNT;
int particleForShape = LOW_RES_COUNT_SHAPE;
float LOW_RES_RADIUS = 1.0f;
float MID_RES_RADIUS = (2.0f / 3.0f);
float HIGH_RES_RADIUS = .50f;
float MAX_RADIUS = LOW_RES_RADIUS;
float SMOOTHING_RADIUS = LOW_RES_RADIUS;
float VISCOSITY = .250f;
float TIMESTEP = .025f;
float MASS = 1.0f;

float FRICTION = .1f;
float ELASTICITY = .7f;
float timePassed = 0.0f;

// surface tension stuff
float TENSION_ALPHA = .25f;
float TENSION_THRESHOLD = 1.0f;
float totalTime = 0.0f;

bool CONTROL = true;

// matchpoint system
vector<Keyframe> keyframes;
vector<Particle> defaultMatchpoints;
const int matchpointNumber = 5;
unsigned int nextKeyframe = 0;
const float permittedError = .01f;

// TODO:
// 1. generate matchpoints (position and radius) at initialization
// 2. Update keyframes at the end of each run
// 3. If recording in mid or high res, do matchpoint error correction


float density_constant = 1.0;
int steps = 0;
int steps_per_update = 3;
Particle* particleList;
Vec3f* particlePositions;

shared_ptr<cy::PointCloud<Vec3f, float, 3>> kdTree;
shared_ptr<cy::TriMesh> shapeMesh;
shared_ptr<cy::BVHTriMesh> bvh;

shared_ptr<Shape> lowResSphere;
std::vector<Plane> surfaces;


glm::vec3 scaleStructure = glm::vec3(.05f, .05f, .05f);
glm::vec3 scaleParticles = glm::vec3(.5f * (resolutionConstant / particleCount), .5f * (resolutionConstant / particleCount), .5f * (resolutionConstant / particleCount));

Scene selected_scene = Scene::SPLASH;

// TODO: fix high resolution transfer

const int N_THREADS = 10;
int PARTICLES_PER_THREAD = LOW_RES_COUNT / N_THREADS;
std::thread threads[N_THREADS];

int width = 1280;
int height = 960;

const char* cmd = "\"C:\\Users\\Sam Hallam\\Desktop\\Art Stuff\\ffmpeg-2021-11-10-git-44c65c6cc0-essentials_build\\bin\\ffmpeg\" -r 30 -f rawvideo -pix_fmt rgba -s 1280x960 -i - "
"-threads 0 -preset fast -y -pix_fmt yuv420p -crf 21 -vf vflip output.mp4";

FILE* ffmpeg = _popen(cmd, "wb");
int* buffer = new int[width * height];

bool recording = false;
float end_time = 0.0f;

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

bool isInBounds(const Vec3f& point, const float* bounds) {
	if (bounds[0] <= point.x && bounds[1] <= point.y && bounds[2] <= point.z) {
		if (bounds[3] >= point.x && bounds[4] >= point.y && bounds[5] >= point.z) {
			return true;
		}
	}

	return false;
}

bool isInBoundingVolume(const Vec3f& point, unsigned int currentNode) {
	if (!isInBounds(point, bvh->GetNodeBounds(currentNode))) {
		return false;
	}
	else if (bvh->IsLeafNode(currentNode)) {
		return true;
	}

	unsigned int child1, child2;
	bvh->GetChildNodes(currentNode, child1, child2);
	return isInBoundingVolume(point, child1) || isInBoundingVolume(point, child2);
}

bool isInBoundingVolume(const Vec3f& point) {
	unsigned int rootNode = bvh->GetRootNodeID();
	return isInBoundingVolume(point, rootNode);
}

void initParticleList_atRest() {
	if (particleList != nullptr) {
		delete[] particleList;
	}
	if (particlePositions != nullptr) {
		delete[] particlePositions;
	}
	particleList = new Particle[particleCount];
	particlePositions = new Vec3f[particleCount + matchpointNumber];

	// put them in a cube shape for ease of access
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
	MASS = volumePerParticle * DENSITY_0_GUESS;
	for (int i = 0; i < depth; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < height; k++) {
				Particle p;
				
				float x_position = ((float)j) * scaleFactor;
				float y_position = ((float)k * .5) * scaleFactor;
				float z_position = (float)i * scaleFactor;
				if (k % 2 == 1) {
					x_position += (.5* scaleFactor);
				}
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

void initParticleShape() {
	Particle* shapeParticles = new Particle[particleCount + particleForShape];
	Vec3f* newPositions = new Vec3f[particleCount + particleForShape + matchpointNumber];

	for (int i = 0; i < particleCount; i++) {
		shapeParticles[i] = particleList[i];
		newPositions[i] = particlePositions[i];
	}

	delete[] particleList;
	delete[] particlePositions;

	// idea: random numbers inside shape
	shapeMesh->ComputeBoundingBox();
	Vec3f min = shapeMesh->GetBoundMin();
	Vec3f max = shapeMesh->GetBoundMax();

	float x_range = max.x - min.x;
	float y_range = max.y - min.y;
	float z_range = max.z - min.z;
	
	float remainingParticles = particleForShape;
	int currentIndex = particleCount;

	// approximate density
	float volume = (4.0 / 3.0) * M_PI * powf(1.75f, 3.0f);
	float density_estimate = (MASS * remainingParticles) / volume;
	//density_estimate *= 100000.f;

	while(remainingParticles > 0) {
		float rand_x = (float)((rand() % 100) + 1.0f) / 100.0f;
		float rand_y = (float)((rand() % 100) + 1.0f) / 100.0f;
		float rand_z = (float)((rand() % 100) + 1.0f) / 100.0f;

		float x = min.x + rand_x * x_range;
		float y = min.y + rand_y * y_range;
		float z = min.z + rand_z * z_range;
		Vec3f potentialParticle = Vec3f(x, y, z);
		
		if (isInBoundingVolume(potentialParticle)) {

			// translate x, y, and z
			x = 3.5 * x + 10;
			y = 3.5 * y + 25;
			z = 3.5 * z + 10;

			Particle p;
			p.setPosition(glm::vec3(x, y, z));
			p.setDensity(density_estimate);
			p.setMass(MASS);
			p.setVelocity(glm::vec3(0.0f, 0.0f, 0.0f));
			p.setRadius(scaleParticles.x);
			potentialParticle.x = x;
			potentialParticle.y = y;
			potentialParticle.z = z;

			shapeParticles[currentIndex] = p;
			newPositions[currentIndex] = potentialParticle;

			remainingParticles--;
			currentIndex++;
		}
	}

	particleList = shapeParticles;
	particlePositions = newPositions;
	particleCount += particleForShape;
	PARTICLES_PER_THREAD = particleCount / N_THREADS;
}

void initParticleList() {
	particleList = new Particle[particleCount];
	particlePositions = new Vec3f[particleCount + matchpointNumber];

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
				float x = j; // +(2 * scaleParticles.x - 1);
				float y = k;
				float z = i; // +(2 * scaleParticles.z - 1);
				p.setPosition(glm::vec3(x, y, z));
				p.setDensity(DENSITY_0_GUESS);
				p.setMass(2.00);
				p.setVelocity(glm::vec3(0.0f, 0.0f, 0.0f));
				p.setRadius(scaleParticles.x);

				int index = (slice * i) + (height * j) + k;
				particleList[index] = p;
				particlePositions[index] = Vec3f(x, y, z);

			}
		}
	}
}

void initSceneOriginal() {
	initParticleList();

	Plane ground(glm::vec3(0.0, 1.0, 0.0), glm::vec3(0.0, -2.0 * scaleParticles.x, 0.0));
	Plane wall_1(glm::vec3(0.0f, 0.0, -1.0), glm::vec3(0.0, 0.0, 21.0f));
	Plane wall_2(glm::vec3(1.0, 0.0, .0), glm::vec3(-1.0, 0.0, 0.0));
	Plane wall_3(glm::vec3(0.0, 0.0, 1.0), glm::vec3(0.0, 0.0, -1.0));
	Plane wall_4(glm::vec3(-1.0, 0.0, 0.0), glm::vec3(21.0f, 0.0, 0.0));

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

	Plane ground(glm::vec3(0.0, 1.0, 0.0), glm::vec3(0.0, 0.0f - .5 , 0.0));
	Plane wall_1(glm::vec3(0.0f, 0.0, -1.0), glm::vec3(0.0, 0.0, 20.0f + .5));
	Plane wall_2(glm::vec3(1.0, 0.0, .0), glm::vec3(0.0 - .5, 0.0f, 0.0));
	Plane wall_3(glm::vec3(0.0, 0.0, 1.0), glm::vec3(0.0, 0.0, 0.0 - .5));
	Plane wall_5(glm::vec3(-1.0, 0.0, 0.0), glm::vec3(40 + .5, 0.0, 0.0));
	Plane wall_4(glm::vec3(-1.0, 0.0, 0.0), glm::vec3(20 + .5, 0.0, 0.0));

	// initialize surfaces
	surfaces.clear();
	surfaces.push_back(ground);
	surfaces.push_back(wall_1);
	surfaces.push_back(wall_2);
	surfaces.push_back(wall_3);
	surfaces.push_back(wall_5);
	surfaces.push_back(wall_4);

}

void initSceneSplash() {
	initParticleList_atRest();
	// HINT: unnatural behavior only occurs when the shape is created for some reason
	// my guess is something wrong with neighbors or something maybe
	initParticleShape();

	Plane ground(glm::vec3(0.0, 1.0, 0.0), glm::vec3(0.0, 0.0f - .5f, 0.0));
	Plane wall_1(glm::vec3(0.0f, 0.0, -1.0), glm::vec3(0.0, 0.0, 20.0f + .5f));
	Plane wall_2(glm::vec3(1.0, 0.0, .0), glm::vec3(0.0 - .5f, 0.0f, 0.0));
	Plane wall_3(glm::vec3(0.0, 0.0, 1.0), glm::vec3(0.0, 0.0, 0.0 - .5f));
	Plane wall_4(glm::vec3(-1.0, 0.0, 0.0), glm::vec3(20 + .5f, 0.0, 0.0));

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
	 if (!kdTree) {
		cout << "Making new kd tree!" << endl;
		kdTree = make_shared<cy::PointCloud<Vec3f, float, 3>>(particleCount + matchpointNumber, particlePositions);
	 }
	else {
		kdTree->Build(particleCount, particlePositions);
	}
}

void initMatchPoints() {
	for (int i = 0; i < matchpointNumber; i++) {
		int particleIndex = rand() % particleCount;

		Particle matchPoint;
		matchPoint.setPosition(particleList[particleIndex].getPosition());
		float radius = rand() % 3 + 1;
		matchPoint.setRadius(radius);
		matchPoint.setMass(MASS);
		Vec3f matchpointPosition = Vec3f(matchPoint.getPosition().x, matchPoint.getPosition().y, matchPoint.getPosition().z);
		particlePositions[particleCount + i] = matchpointPosition;
		matchPoint.setIsMatchpoint(true);

		defaultMatchpoints.push_back(matchPoint);
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
	
	keyToggles[(unsigned)'l'] = true;

	// initialize shape for extra mesh, if desired
	// TODO: can add other shapes later. for now spheres only baby
	shapeMesh = make_shared<cy::TriMesh>();
	std::string mesh_location = RESOURCE_DIR + "low_res_sphere.obj";
	if (!shapeMesh->LoadFromFileObj(mesh_location.c_str(), false)) {
		cout << "Error loading mesh into triangle mesh" << endl;
	}
	shapeMesh->ComputeBoundingBox();

	// initializes bounding volume with shape
	bvh = make_shared<cy::BVHTriMesh>(shapeMesh.get());


	// initialize particles and tree
	switch (selected_scene) {
		case Scene::DAM_BREAK:
			initSceneDamBreak();
			break;
		case Scene::SPLASH:
			initSceneSplash();
			break;
		default:
			initSceneOriginal();
	}
	initMatchPoints();
	initKdTree();
	

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
	glUniform3f(prog->getUniform("lightPos"), 20.0f, 20.0f, -20.0f);
	MV->scale(scaleStructure);
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

float calculateDensityForParticle(const Particle x) {
	float density = x.getMass() * Kernel::polyKernelFunction(x, x, x.getIsMatchPoint());
	for (int j = 0; j < x.getNeighbors().size(); j++) {
		Particle* xj = x.getNeighbors().at(j);
		density += (xj->getMass() * Kernel::polyKernelFunction(x, *xj, x.getIsMatchPoint()));
	}

	return (density * density_constant);
}

float sampleDensityForMatchpoint(const Particle x) {
	float sample = 0;
	float normalizer = 0;
	for (int j = 0; j < x.getNeighbors().size(); j++) {
		Particle* xj = x.getNeighbors().at(j);
		sample += (xj->getDensity() * Kernel::samplingKernel(x, *xj, x.getIsMatchPoint()));
		normalizer += (Kernel::samplingKernel(x, *xj, x.getIsMatchPoint()));
	}
	// TODO: division???

	return sample / normalizer;
}

float calculatePressureForParticle(const Particle x) {
	//float pressure = ((STIFFNESS_PARAM * DENSITY_0_GUESS) / Y_PARAM) *(powf((x.getDensity() / DENSITY_0_GUESS), Y_PARAM) - 1.0f);
	float pressure = STIFFNESS_PARAM * (x.getDensity() - DENSITY_0_GUESS);
	return pressure;
}

glm::vec3 pressureGradient(const Particle& xi) {
	glm::vec3 pressureGradient = glm::vec3(0.0f, 0.0f, 0.0f);

	// for every Particle xj in the neighbor hood of xi
	for (int j = 0; j < xi.getNeighbors().size(); j++) {
		Particle* xj = xi.getNeighbors().at(j);

		//float pressureTerm = (xi.getPressure() / powf(xi.getDensity(), 2.0f)) + (xj->getPressure() / powf(xj->getDensity(), 2.0f));
		
		float pressureTerm = (xi.getPressure() + xj->getPressure()) / (2 * xj->getDensity());
		pressureGradient += (xj->getMass() * pressureTerm * Kernel::spikyKernelGradient(xi, *xj));
	}
	return -1.0f * pressureGradient;
}

void setNeighbors(Particle& x, int xIndex) {
	float radius = x.getIsMatchPoint() ? x.getRadius() : MAX_RADIUS;
	cy::PointCloud<Vec3f, float, 3>::PointInfo* info = new cy::PointCloud<Vec3f, float, 3>::PointInfo [500];
	int numPointsInRadius = kdTree->GetPoints(particlePositions[xIndex], MAX_RADIUS, 500, info);

	// create a vector for the new neighbors
	std::vector<Particle*> neighbors;
	for (int i = 0; i < numPointsInRadius; i++) {
		if (xIndex != info[i].index && !particleList[info[i].index].getIsMatchPoint()) {
			neighbors.push_back(&particleList[info[i].index]);
		}
	}
	
	x.setNeighbors(neighbors);
	delete[] info;
}

glm::vec3 diffusionTerm(const Particle& xi) {
	glm::vec3 diffusionLaplacian = glm::vec3(0.0f, 0.0f, 0.0f);

	// for every Particle xj in the neighbor hood of xi
	for (int j = 0; j < xi.getNeighbors().size(); j++) {
		Particle* xj = xi.getNeighbors().at(j);
		glm::vec3 velocityTerm = (xj->getVelocity() - xi.getVelocity()) / xj->getDensity();

		diffusionLaplacian += (xj->getMass() * velocityTerm * Kernel::viscosityKernelLaplacian(xi, *xj));
	}
	return diffusionLaplacian * VISCOSITY;
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
float colorFieldLaplacian(const Particle& xi) {
	float surfaceField = 0;

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

void setDensitiesForParticles(int start_index, int end_index) {
	for (int i = start_index; i < end_index; i++) {
		particleList[i].setDensity(calculateDensityForParticle(particleList[i]));
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

void setAccelerationForParticles(int start_index, int end_index) {
	for (int i = start_index; i < end_index; i++) {
		glm::vec3 pressureForce = pressureGradient(particleList[i]);
		glm::vec3 diffusionForce = diffusionTerm(particleList[i]);
		glm::vec3 externalForce = glm::vec3(0.0, -9.8f * particleList[i].getDensity(), 0.0f);

		// calculate surface pressure/tension
		
		glm::vec3 acceleration = pressureForce + diffusionForce + externalForce;

		if (TENSION_ALPHA > 0.0){
			float k = -1.0f * particleList[i].getColorFieldLaplacian() / length(particleList[i].getSurfaceNormal());
			glm::vec3 tension = k * TENSION_ALPHA * particleList[i].getSurfaceNormal();
			if (length(tension) > TENSION_THRESHOLD) {
				acceleration += tension;
			}
		}

		acceleration /= particleList[i].getDensity();
		particleList[i].setAcceleration(acceleration);
	}
}

void updatePositionForParticles(int start_index, int end_index, double time) {
	for (int i = start_index; i < end_index; i++) {
		float timeStepRemaining = time;
		glm::vec3 newVelocity = particleList[i].getVelocity() + particleList[i].getAcceleration() * timeStepRemaining;
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

void updateMatchPoints(float time) {
	Keyframe k;
	k.time = time + timePassed;

	// TODO: add the matchpoints to tree and calculate each trait of match point
	for (int i = 0; i < matchpointNumber; i++) {
		particlePositions[particleCount + i] = Vec3f(defaultMatchpoints.at(i).getPosition().x, defaultMatchpoints.at(i).getPosition().y, defaultMatchpoints.at(i).getPosition().z);
		// set the neighbors for match point
		setNeighbors(defaultMatchpoints.at(i), particleCount + i);

		// calculate density for matchpoint
		defaultMatchpoints.at(i).setDensity(sampleDensityForMatchpoint(defaultMatchpoints.at(i)));

		// add the match point to the keyframe
		k.matchpoints.push_back(defaultMatchpoints.at(i));
	}

	keyframes.push_back(k);
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

	// update density
	for (int i = 0; i < N_THREADS; i++) {
		// particleList[i].setDensity(calculateDensityForParticle(particleList[i]));
		threads[i] = thread(setDensitiesForParticles, i * PARTICLES_PER_THREAD, (i + 1) * PARTICLES_PER_THREAD);
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

	// do key frame stuff
	if (!recording) {
		updateMatchPoints(time);
	}
	else if(CONTROL) {
		// apply match point control
		// 0. Figure out if we're at a keyframe time
		if (nextKeyframe < keyframes.size() && keyframes.at(nextKeyframe).time == (timePassed + time)) {
			// loop through matchpoints
			int iterations = 0;
			for (int i = 0; i < matchpointNumber; i++) {
				Particle matchpoint = keyframes.at(nextKeyframe).matchpoints.at(i);

				// 1. Sample high resolution model at point
				Particle highResSample;
				highResSample.setPosition(keyframes.at(nextKeyframe).matchpoints.at(i).getPosition());
				highResSample.setRadius(keyframes.at(nextKeyframe).matchpoints.at(i).getRadius());
				highResSample.setMass(keyframes.at(nextKeyframe).matchpoints.at(i).getMass());

				// get high res sample neighbors
				particlePositions[particleCount + i] = Vec3f(highResSample.getPosition().x, highResSample.getPosition().y, highResSample.getPosition().z);
				setNeighbors(highResSample, particleCount + i);

				// calculate high res density
				highResSample.setDensity(sampleDensityForMatchpoint(highResSample));

				// 2. Calculate error between high and low res value
				float densityError = matchpoint.getMass() * (matchpoint.getDensity() - highResSample.getDensity());
				float absError = abs(((matchpoint.getDensity() - highResSample.getDensity()) / matchpoint.getDensity()));

				while (absError > permittedError || iterations < 6) {
					// 3. Calcuate G'(r, x)
					float totalError = 0;
					for (int j = 0; j < highResSample.getNeighbors().size(); j++) {
						Particle* xj = highResSample.getNeighbors().at(j);
						float kernel = Kernel::samplingKernel(highResSample, *xj, highResSample.getIsMatchPoint());
						totalError += powf(kernel, 2.0f);
					}

					// 4. Apply control for each neighbor
					for (int j = 0; j < highResSample.getNeighbors().size(); j++) {
						Particle* xj = highResSample.getNeighbors().at(j);
						float kernel = Kernel::samplingKernel(highResSample, *xj, highResSample.getIsMatchPoint());
						
						float newDensity = xj->getDensity() + densityError * (kernel / totalError);
						xj->setDensity(newDensity);
					}
					
					// update sampled density and error
					highResSample.setDensity(sampleDensityForMatchpoint(highResSample));

					// update error
					float densityError = matchpoint.getMass() * (matchpoint.getDensity() - highResSample.getDensity());
					float absError = abs(((matchpoint.getDensity() - highResSample.getDensity()) / matchpoint.getDensity()));
					iterations++;
				}
			}

			nextKeyframe++;
		}
	}

	steps++;
}

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
			ImGui::SliderFloat("\"Y\" Parameter", &Y_PARAM, 0.0f, 10.0f);
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
		if (ImGui::Button("Record in High Resolution")) {
			recording = true;
			
			end_time = timePassed;
			timePassed = 0.0f;
			// FIXME: leftover particles after switch
			kdTree = nullptr;
			particleCount = HIGH_RES_COUNT;
			particleForShape = HIGH_RES_COUNT_SHAPE;
			SMOOTHING_RADIUS = HIGH_RES_RADIUS;
			Kernel::setSmoothingRadius(SMOOTHING_RADIUS);
			MAX_RADIUS = HIGH_RES_RADIUS;
			float scaleFactor = (powf(resolutionConstant, (1.f / 3.f)) / powf(particleCount, (1.f / 3.f)));
			scaleParticles = glm::vec3(.5f * (scaleFactor), .5f * (scaleFactor), .5f * (scaleFactor));
			PARTICLES_PER_THREAD = HIGH_RES_COUNT / N_THREADS;
			TIMESTEP = .01f;
			DENSITY_0_GUESS = DENSITY_0_GUESS / scaleFactor;

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
		}
		if (ImGui::Button("Record in Mid Resolution")) {
			recording = true;

			end_time = timePassed;
			timePassed = 0.0f;
			// FIXME: leftover particles after switch
			kdTree = nullptr;
			particleCount = MID_RES_COUNT;
			particleForShape = MID_RES_COUNT_SHAPE;
			SMOOTHING_RADIUS = MID_RES_RADIUS;
			Kernel::setSmoothingRadius(SMOOTHING_RADIUS);
			MAX_RADIUS = MID_RES_RADIUS;
			float scaleFactor = (powf(resolutionConstant, (1.f / 3.f)) / powf(particleCount, (1.f / 3.f)));
			scaleParticles = glm::vec3(.5f * (scaleFactor), .5f * (scaleFactor), .5f * (scaleFactor));
			PARTICLES_PER_THREAD = particleCount / N_THREADS;
			TIMESTEP = .01f;
			DENSITY_0_GUESS = DENSITY_0_GUESS / scaleFactor;

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

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 460");

	std::string buttonText = "Play";
	bool isPaused = true;
	cout << "Number of surfaces: " << surfaces.size() << endl;
	while(!glfwWindowShouldClose(window) || (recording && timePassed <= end_time)) {
		if(!glfwGetWindowAttrib(window, GLFW_ICONIFIED)) {
			
			// Simulate and draw water
			if (!isPaused) {
				float timeEnd = glfwGetTime();
				// Integrate partices
				updateFluid(TIMESTEP);
				// Render scene.
				timePassed += (TIMESTEP);
				totalTime += timePassed;
				if (timePassed >= 6 && surfaces.size() == 6 && selected_scene == Scene::DAM_BREAK) {
					cout << "Release the kracken!" << endl;
					surfaces.resize(5);
				}
				GLSL::checkError(GET_FILE_LINE);
			}
			
			render();
			if (!(recording && timePassed <= end_time)) {
				renderGui(isPaused, buttonText);
			}
			if (recording && timePassed > end_time) {
				cout << "Recording finished" << endl;
				break;
			}
			
			// Swap front and back buffers.
			glfwSwapBuffers(window);

			if (recording) {
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
	delete[] particleList;

	// Quit program.
	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}
