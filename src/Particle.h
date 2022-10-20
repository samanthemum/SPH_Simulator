#pragma once
#ifndef _PARTICLE_H
#define _PARTICLE_H

#include <memory>
#include <glm/common.hpp>
#include <glm/fwd.hpp>
#include "Plane.h"

class Particle {
	public:
		Particle() {};
		~Particle() {};

		struct Forces {
			glm::vec3 pressure{ 0, 0, 0 };
			glm::vec3 diffusion{ 0 };
			glm::vec3 external{ 0 };
		};

		// setter functions
		void setPosition(glm::vec3 pos) { this->position = pos; };
		void setVelocity(glm::vec3 v) { this->velocity = v; };
		void setAcceleration(glm::vec3 a) { this->acceleration = a; };
		void setSurfaceNormal(glm::vec3 newNormal) { this->surfaceNormal = newNormal; };
		void setColorFieldLaplacian(glm::vec3 newColor) { this->colorFieldLaplacian = newColor; };
		void setDensity(float density) { this->density = density; };
		void setDensityError(float density_e) { this->density_error = density_e; };
		void setPressure(float p) { this->pressure = p; };
		void setMass(float m) { this->mass = m; };
		void setVolume(float v) { this->volume = v; };
		void setRadius(float r) { this->radius = r; }
		void setNeighbors(std::vector<Particle*> n) { this->neighbors = n; }
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
		glm::vec3 getPosition() const { return position; };
		glm::vec3 getVelocity() const { return velocity; };
		glm::vec3 getAcceleration() const { return acceleration; };
		glm::vec3 getSurfaceNormal() const { return surfaceNormal; }
		glm::vec3 getColorFieldLaplacian() const { return colorFieldLaplacian; }
		float getDensity() const { return density; };
		float getDensityError() const { return density_error; };
		float getPressure() const { return pressure; };
		float getMass() const { return mass; };
		float getVolume() const { return volume; };
		float getRadius() const { return radius; }
		std::vector<Particle*> getNeighbors() const { return neighbors; }

		// refactor later
		Forces forces;
		glm::vec3 position_predicted;
		glm::vec3 velocity_predicted = glm::vec3(0.0f, 0.0f, 0.0f);

	private:
		glm::vec3 position;
		glm::vec3 velocity = glm::vec3(0.0f, 0.0f, 0.0f);
		glm::vec3 acceleration;
		glm::vec3 surfaceNormal;
		glm::vec3 colorFieldLaplacian;
		std::vector<Particle*> neighbors;
		float density;
		float density_error = 0.0f;
		float mass;
		float radius;
		float volume;
		float pressure;
		
};
#endif