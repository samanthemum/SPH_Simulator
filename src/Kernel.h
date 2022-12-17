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
		static float polyKernelFunction(const Particle& xi, const Particle& xj, bool useCustomRadius = false) {
			// if we're less than the max radius, don't do anything
			float radius = useCustomRadius ? xi.getRadius() : SMOOTHING_RADIUS;

			if (length(xi.getPosition() - xj.getPosition()) > radius) {
				return 0;
			}

			// otherwise
			float outsideTerm = 315.0f / (M_PI * powf(radius, 9.0f) * 64.0f);
			float insideTerm = powf(powf(radius, 2.0f) - powf(length(xi.getPosition() - xj.getPosition()), 2.0f), 3.0f);
			return outsideTerm * insideTerm;
		}

		static float samplingKernel(const Particle& xi, const Particle& xj, bool useCustomRadius = false) {
			// if we're less than the max radius, don't do anything
			float radius = useCustomRadius ? xi.getRadius() : SMOOTHING_RADIUS;

			if (length(xi.getPosition() - xj.getPosition()) > radius) {
				return 0;
			}

			// otherwise
			float kernelValue = exp(-1 * radius * powf(length(xi.getPosition() - xj.getPosition()), 2.0f));
			return kernelValue;
		}

		static glm::vec3 spikyKernelGradient(const Particle& xi, const Particle& xj) {
			// if we're less than the max radius, don't do anything
			if (length(xi.getPosition() - xj.getPosition()) > SMOOTHING_RADIUS) {
				return glm::vec3(0,0,0);
			}

			// TODO: find out why we're self including to begin with
			if (length(xi.getPosition() - xj.getPosition()) == 0.0f) {
				return glm::vec3(0, 0, 0);
			}

			// otherwise
			float outsideTerm = -45.0f / (M_PI * powf(SMOOTHING_RADIUS, 6.0f));
			float insideTerm = powf(SMOOTHING_RADIUS - length(xi.getPosition() - xj.getPosition()), 2.0f);
			return outsideTerm * insideTerm * normalize(xi.getPosition() - xj.getPosition());
		}

		// TODO: fix this function
		static float polyKernelLaplacian(const Particle& xi, const Particle& xj) {
			// if we're less than the max radius, don't do anything
			if (length(xi.getPosition() - xj.getPosition()) > SMOOTHING_RADIUS) {
				return 0;
			}

			// otherwise
			glm::vec3 r = xi.getPosition() - xj.getPosition();
			float outsideTerm = 3.0f * 315.0f * length(r) * length(r) / (M_PI * powf(SMOOTHING_RADIUS, 9.0f) * 8.0f);
			float insideTerm = powf(SMOOTHING_RADIUS, 2.0f) - powf(length(xi.getPosition() - xj.getPosition()), 2.0f);
			// glm::vec3 vectorTerm = glm::vec3((powf(r.y, 2.0f) + powf(r.z, 2.0f)) / powf(length(r), 3.0f), (powf(r.x, 2.0f) + powf(r.z, 2.0f)) / powf(length(r), 3.0f), (powf(r.y, 2.0f) + powf(r.x, 2.0f)) / powf(length(r), 3.0f));
			return outsideTerm * insideTerm;
		}


		static float viscosityKernelLaplacian(const Particle& xi, const Particle& xj) {
			// if we're less than the max radius, don't do anything
			if (length(xi.getPosition() - xj.getPosition()) > SMOOTHING_RADIUS) {
				return 0;
			}

			float outsideTerm = 45.0f / (M_PI * powf(SMOOTHING_RADIUS, 6.0f));
			float insideTerm = SMOOTHING_RADIUS - length(xi.getPosition() - xj.getPosition());
			return outsideTerm * insideTerm;
		}

		static glm::vec3 polyKernelGradient(const Particle& xi, const Particle& xj) {
			// if we're less than the max radius, don't do anything
			if (length(xi.getPosition() - xj.getPosition()) > SMOOTHING_RADIUS) {
				return glm::vec3(0.0f, 0.0f, 0.0f);
			}

			// otherwise
			glm::vec3 r = xi.getPosition() - xj.getPosition();
			float outsideTerm = -3.0f * 315.0f * length(xi.getPosition() - xj.getPosition()) / (M_PI * powf(SMOOTHING_RADIUS, 9.0f) * 32.0f);
			float insideTerm = powf(powf(SMOOTHING_RADIUS, 2.0f) - powf(length(xi.getPosition() - xj.getPosition()), 2.0f), 2.0f);
			glm::vec3 vectorTerm = glm::vec3(r.x / length(r), r.y / length(r), r.z / length(r));
			return outsideTerm * insideTerm * vectorTerm;
		}

	
};

float Kernel::SMOOTHING_RADIUS = 1.0f;
#endif