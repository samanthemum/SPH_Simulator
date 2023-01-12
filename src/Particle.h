#pragma once
#ifndef _PARTICLE_H
#define _PARTICLE_H

#include <memory>
#include <glm/common.hpp>
#include <glm/fwd.hpp>
#include <glm/glm.hpp>
#include <vector>
#include "Plane.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

class Particle {
	public:
		CUDA_CALLABLE_MEMBER Particle() {};
		CUDA_CALLABLE_MEMBER ~Particle() {
		};

		// setter functions
		CUDA_CALLABLE_MEMBER void setPosition(glm::vec3 pos) { this->position = pos; };
		CUDA_CALLABLE_MEMBER void setVelocity(glm::vec3 v) { this->velocity = v; };
		CUDA_CALLABLE_MEMBER void setAcceleration(glm::vec3 a) { this->acceleration = a; };
		CUDA_CALLABLE_MEMBER void setSurfaceNormal(glm::vec3 newNormal) { this->surfaceNormal = newNormal; };
		CUDA_CALLABLE_MEMBER void setColorFieldLaplacian(float newColor) { this->colorFieldLaplacian = newColor; };
		CUDA_CALLABLE_MEMBER void setDensity(float density) { this->density = density; };
		CUDA_CALLABLE_MEMBER void setPressure(float p) { this->pressure = p; };
		void setMass(float m) { this->mass = m; };
		void setVolume(float v) { this->volume = v; };
		void setRadius(float r) { this->radius = r; }
		void setNeighbors(std::vector<Particle*> n) { this->neighbors = n; }
		// void setNeighborIndices(std::vector<int> n) { this->neighborIndices = n; }
		void setIsMatchpoint(bool newMatchPointVal) { this->isMatchPoint = newMatchPointVal; }
		static bool willCollideWithPlane(glm::vec3 position, glm::vec3 newPos, float radius, const Plane& p) {
			float oldDistance;
			if (glm::dot((position - p.getPoint()), p.getNormal()) >= 0.0f) {
				oldDistance = glm::dot((position - p.getPoint()), p.getNormal()) - radius;
			}
			else {
				oldDistance = glm::dot((position - p.getPoint()), p.getNormal()) + radius;
			}

			float newDistance;
			if (glm::dot((newPos - p.getPoint()), p.getNormal()) >= 0.0f) {
				newDistance = glm::dot((newPos - p.getPoint()), p.getNormal()) - radius;
			}
			else {
				newDistance = glm::dot((newPos - p.getPoint()), p.getNormal()) + radius;
			}

			if (glm::dot((newPos - p.getPoint()), p.getNormal()) > 0.0f && glm::dot((position - p.getPoint()), p.getNormal()) <= 0.0f) {
				return true;
			}

			if (glm::dot((newPos - p.getPoint()), p.getNormal()) < 0.0f && glm::dot((position - p.getPoint()), p.getNormal()) >= 0.0f) {
				return true;
			}

			// same signedness == no collisions
			if ((newDistance > 0 && oldDistance > 0) || (newDistance < 0 && oldDistance < 0) || (newDistance == 0 && oldDistance == 0)) {
				return false;
			}

			return true;
		}

		static float getDistanceFromPlane(glm::vec3 position, float radius, const Plane& p) {
			float distance;
			if (glm::dot((position - p.getPoint()), p.getNormal()) >= 0.0f) {
				distance = glm::dot((position - p.getPoint()), p.getNormal()) - radius;
			}
			else {
				distance = glm::dot((position - p.getPoint()), p.getNormal()) + radius;
			}

			return abs(distance);
		}
		
		// getter functions
		CUDA_CALLABLE_MEMBER glm::vec3 getPosition() const { return position; };
		CUDA_CALLABLE_MEMBER glm::vec3 getVelocity() const { return velocity; };
		CUDA_CALLABLE_MEMBER glm::vec3 getAcceleration() const { return acceleration; };
		CUDA_CALLABLE_MEMBER glm::vec3 getSurfaceNormal() const { return surfaceNormal; }
		CUDA_CALLABLE_MEMBER float getColorFieldLaplacian() const { return colorFieldLaplacian; }
		CUDA_CALLABLE_MEMBER float getDensity() const { return density; };
		CUDA_CALLABLE_MEMBER float getPressure() const { return pressure; };
		CUDA_CALLABLE_MEMBER float getMass() const { return mass; };
		float getVolume() const { return volume; };
		float getRadius() const { return radius; }
		std::vector<Particle*> getNeighbors() const { return neighbors; }
		CUDA_CALLABLE_MEMBER bool getIsMatchPoint() const { return isMatchPoint; }
		static const int maxNeighborsAllowed = 200;
		int* neighborIndices = nullptr;
		int numNeighbors;

	private:
		glm::vec3 position;
		glm::vec3 velocity = glm::vec3(0.0f, 0.0f, 0.0f);
		glm::vec3 acceleration;
		glm::vec3 surfaceNormal;
		float colorFieldLaplacian;
		std::vector<Particle*> neighbors;
		// std::vector<int> neighborIndices;
		float density;
		float mass;
		float radius;
		float volume;
		float pressure;
		bool isMatchPoint = false;
};
#endif