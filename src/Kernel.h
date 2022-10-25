#pragma once
#ifndef KERNEL_H_
#define KERNEL_H_

#include "Particle.h"
#include "Plane.h"
#include <glm/common.hpp>
#include <glm/fwd.hpp>

class Kernel {
	private:
		static float SMOOTHING_RADIUS;

	public:
		static void setSmoothingRadius(float radius) { SMOOTHING_RADIUS = radius; }
		static float polyKernelFunction(const Particle& xi, const Particle& xj, bool predicted = false) {
			// if we're less than the max radius, don't do anything
			glm::vec3 r;
			if (predicted) {
				r = xi.getPredictedPosition() - xj.getPredictedPosition();
			}
			else {
				r = xi.getPosition() - xj.getPosition();
			}

			if (length(r) > SMOOTHING_RADIUS) {
				return 0;
			}

			// otherwise
			float outsideTerm = 315.0f / (M_PI * powf(SMOOTHING_RADIUS, 9.0f) * 64.0f);
			float insideTerm = powf(powf(SMOOTHING_RADIUS, 2.0f) - powf(length(r), 2.0f), 3.0f);
			return outsideTerm * insideTerm;
		}


		static float spikyKernelGradient(const Particle& xi, const Particle& xj, bool predicted = false) {
			// if we're less than the max radius, don't do anything
			glm::vec3 r;
			if (predicted) {
				r = xi.getPredictedPosition() - xj.getPredictedPosition();
			}
			else {
				r = xi.getPosition() - xj.getPosition();
			}

			if (length(r) > SMOOTHING_RADIUS) {
				return 0;
			}

			// otherwise
			float outsideTerm = -45.0f / (M_PI * powf(SMOOTHING_RADIUS, 6.0f));
			float insideTerm = powf(SMOOTHING_RADIUS - length(r), 2.0f);
			return outsideTerm * insideTerm;
		}

		static glm::vec3 polyKernelLaplacian(const Particle& xi, const Particle& xj, bool predicted = false) {
			// if we're less than the max radius, don't do anything
			glm::vec3 r;
			if (predicted) {
				r = xi.getPredictedPosition() - xj.getPredictedPosition();
			}
			else {
				r = xi.getPosition() - xj.getPosition();
			}

			if (length(r) > SMOOTHING_RADIUS) {
				return glm::vec3(0.0f, 0.0f, 0.0f);
			}

			// otherwise
			float outsideTerm = 3.0f * 315.0f * length(r) * length(r) / (M_PI * powf(SMOOTHING_RADIUS, 9.0f) * 8.0f);
			float insideTerm = powf(SMOOTHING_RADIUS, 2.0f) - powf(length(r), 2.0f);
			glm::vec3 vectorTerm = glm::vec3((powf(r.y, 2.0f) + powf(r.z, 2.0f)) / powf(length(r), 3.0f), (powf(r.x, 2.0f) + powf(r.z, 2.0f)) / powf(length(r), 3.0f), (powf(r.y, 2.0f) + powf(r.x, 2.0f)) / powf(length(r), 3.0f));
			return outsideTerm * insideTerm * vectorTerm;
		}


		static float viscosityKernelLaplacian(const Particle& xi, const Particle& xj, bool predicted = false) {
			// if we're less than the max radius, don't do anything
			glm::vec3 r;
			if (predicted) {
				r = xi.getPredictedPosition() - xj.getPredictedPosition();
			}
			else {
				r = xi.getPosition() - xj.getPosition();
			}

			if (length(r) > SMOOTHING_RADIUS) {
				return 0;
			}
			float outsideTerm = 45.0f / (M_PI * powf(SMOOTHING_RADIUS, 6.0f));
			float insideTerm = SMOOTHING_RADIUS - length(r);
			return outsideTerm * insideTerm;
		}

		static glm::vec3 polyKernelGradient(const Particle& xi, const Particle& xj, bool predicted = false) {
			// if we're less than the max radius, don't do anything
			glm::vec3 r;
			if (predicted) {
				r = xi.getPredictedPosition() - xj.getPredictedPosition();
			}
			else {
				r = xi.getPosition() - xj.getPosition();
			}

			if (length(r) > SMOOTHING_RADIUS) {
				return glm::vec3(0.0f, 0.0f, 0.0f);
			}

			// otherwise
			float outsideTerm = -3.0f * 315.0f * length(r) / (M_PI * powf(SMOOTHING_RADIUS, 9.0f) * 32.0f);
			float insideTerm = powf(powf(SMOOTHING_RADIUS, 2.0f) - powf(length(r), 2.0f), 2.0f);
			glm::vec3 vectorTerm = glm::vec3(r.x / length(r), r.y / length(r), r.z / length(r));
			return outsideTerm * insideTerm * vectorTerm;
		}

	
};

float Kernel::SMOOTHING_RADIUS = 1.0f;
#endif