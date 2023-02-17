#pragma once
#ifndef KERNEL_H_
#define KERNEL_H_

#define _USE_MATH_DEFINES
#include <math.h>

#include "Particle.h"
#include "Plane.h"
#include <glm/common.hpp>
#include <glm/fwd.hpp>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 


class Kernel {
	private:
		float SMOOTHING_RADIUS = 1.0;

	public:
		Kernel() {};
		~Kernel() {};
		CUDA_CALLABLE_MEMBER float getSmoothingRadius() { return SMOOTHING_RADIUS; }
		void setSmoothingRadius(float radius) { SMOOTHING_RADIUS = radius; }
		CUDA_CALLABLE_MEMBER float polyKernelFunction(const Particle& xi, const Particle& xj, bool useCustomRadius = false) {
			// if we're less than the max radius, don't do anything
			float radius = useCustomRadius ? xi.getRadius() : getSmoothingRadius();

			if (length(xi.getPosition() - xj.getPosition()) > radius) {
				return 0;
			}

			// if somehow two particles have the exact same position, run
			/*if (length(xi.getPosition() - xj.getPosition()) == 0.0f) {
				return 0;
			}*/

			// otherwise
			float outsideTerm = 315.0f / (M_PI * powf(radius, 9.0f) * 64.0f);
			float insideTerm = powf(powf(radius, 2.0f) - powf(length(xi.getPosition() - xj.getPosition()), 2.0f), 3.0f);
			return outsideTerm * insideTerm;
		}

		float samplingKernel(const Particle& xi, const Particle& xj, bool useCustomRadius = false) {
			// if we're less than the max radius, don't do anything
			float radius = useCustomRadius ? xi.getRadius() : getSmoothingRadius();

			/*if (length(xi.getPosition() - xj.getPosition()) > radius) {
				return 0;
			}*/

			// otherwise
			float kernelValue = exp(-1.f * radius * powf(length(xi.getPosition() - xj.getPosition()), 2.0f));
			return kernelValue;
		}

		CUDA_CALLABLE_MEMBER glm::vec3 spikyKernelGradient(const Particle& xi, const Particle& xj) {
			// if we're less than the max radius, don't do anything
			if (length(xi.getPosition() - xj.getPosition()) > getSmoothingRadius()) {
				return glm::vec3(0,0,0);
			}

			// TODO: find out why we're self including to begin with
			if (length(xi.getPosition() - xj.getPosition()) == 0.0f) {
				return glm::vec3(0, 0, 0);
			}

			// otherwise
			float outsideTerm = -45.0f / (M_PI * powf(getSmoothingRadius(), 6.0f));
			float insideTerm = powf(getSmoothingRadius() - length(xi.getPosition() - xj.getPosition()), 2.0f);
			return outsideTerm * insideTerm * normalize(xi.getPosition() - xj.getPosition());
		}

		// TODO: fix this function
		CUDA_CALLABLE_MEMBER float polyKernelLaplacian(const Particle& xi, const Particle& xj) {
			// if we're less than the max radius, don't do anything
			if (length(xi.getPosition() - xj.getPosition()) > getSmoothingRadius()) {
				return 0;
			}

			if (length(xi.getPosition() - xj.getPosition()) == 0.0f) {
				return 0;
			}

			// otherwise
			glm::vec3 r = xi.getPosition() - xj.getPosition();
			float outsideTerm = 3.0f * 315.0f * length(r) * length(r) / (M_PI * powf(getSmoothingRadius(), 9.0f) * 8.0f);
			float insideTerm = powf(getSmoothingRadius(), 2.0f) - powf(length(xi.getPosition() - xj.getPosition()), 2.0f);
			// glm::vec3 vectorTerm = glm::vec3((powf(r.y, 2.0f) + powf(r.z, 2.0f)) / powf(length(r), 3.0f), (powf(r.x, 2.0f) + powf(r.z, 2.0f)) / powf(length(r), 3.0f), (powf(r.y, 2.0f) + powf(r.x, 2.0f)) / powf(length(r), 3.0f));
			return outsideTerm * insideTerm;
		}


		CUDA_CALLABLE_MEMBER float viscosityKernelLaplacian(const Particle& xi, const Particle& xj) {
			// if we're less than the max radius, don't do anything
			if (length(xi.getPosition() - xj.getPosition()) > getSmoothingRadius()) {
				return 0;
			}

			if (length(xi.getPosition() - xj.getPosition()) == 0.0f) {
				return 0;
			}

			float outsideTerm = 45.0f / (M_PI * powf(getSmoothingRadius(), 6.0f));
			float insideTerm = getSmoothingRadius() - length(xi.getPosition() - xj.getPosition());
			return outsideTerm * insideTerm;
		}

		CUDA_CALLABLE_MEMBER glm::vec3 polyKernelGradient(const Particle& xi, const Particle& xj) {
			// if we're less than the max radius, don't do anything
			if (length(xi.getPosition() - xj.getPosition()) > getSmoothingRadius()) {
				return glm::vec3(0.0f, 0.0f, 0.0f);
			}

			if (length(xi.getPosition() - xj.getPosition()) == 0.0f) {
				return glm::vec3(0, 0, 0);
			}

			// otherwise
			glm::vec3 r = xi.getPosition() - xj.getPosition();
			float outsideTerm = -3.0f * 315.0f * length(xi.getPosition() - xj.getPosition()) / (M_PI * powf(getSmoothingRadius(), 9.0f) * 32.0f);
			float insideTerm = powf(powf(getSmoothingRadius(), 2.0f) - powf(length(xi.getPosition() - xj.getPosition()), 2.0f), 2.0f);
			glm::vec3 vectorTerm = glm::vec3(r.x / length(r), r.y / length(r), r.z / length(r));
			return outsideTerm * insideTerm * vectorTerm;
		}

		float monaghanKernel(const Particle& xi, const Particle& xj, bool useCustomRadius = false, bool predicted = false) {
			glm::vec3 r;
			float radius = useCustomRadius ? xi.getRadius() : getSmoothingRadius();
			/*if (predicted) {
				r = xi.getPredictedPosition() - xj.getPredictedPosition();
			}*/
			//else {
				r = xi.getPosition() - xj.getPosition();
			//}

			if (length(r) > 2 * radius) {
				return 0;
			}

			float outsideTerm = 1 / (M_PI * powf(radius, 3.0f));
			float insideTerm;
			if (length(r) > radius) {
				insideTerm = .25f * powf(2 - (length(r) / radius), 3.0f);
			}
			else {
				insideTerm = 1.f - (3.f / 2.f) * powf(length(r) / radius, 2.f) + (3.f / 4.f) * powf(length(r) / radius, 3.f);
			}

			// otherwise
			return outsideTerm * insideTerm;
		}

		glm::vec3 monaghanKernelGradient(const Particle& xi, const Particle& xj, bool useCustomRadius = false, bool predicted = false) {
			glm::vec3 r;
			float radius = useCustomRadius ? xi.getRadius() : getSmoothingRadius();
			/*if (predicted) {
				r = xi.getPredictedPosition() - xj.getPredictedPosition();
			}*/
			//else {
				r = xi.getPosition() - xj.getPosition();
			//}

			if (length(r) > 2 * radius) {
				return glm::vec3(0, 0, 0);
			}

			float outsideTerm = 1 / (M_PI * powf(radius, 4.0f));
			float insideTerm;
			if (length(r) > radius) {
				insideTerm = -.75f * powf(2 - (length(r) / radius), 2.0f);
			}
			else {
				insideTerm = 3 * (length(r) / radius) * (-1 + .75 * (length(r) / radius));
			}

			// otherwise
			return outsideTerm * insideTerm * normalize(r);
		}

		float monaghanKernelLaplacian(const Particle& xi, const Particle& xj, bool useCustomRadius = false, bool predicted = false) {
			glm::vec3 r;
			float radius = useCustomRadius ? xi.getRadius() : getSmoothingRadius();
			/*if (predicted) {
				r = xi.getPredictedPosition() - xj.getPredictedPosition();
			}*/
			//else {
				r = xi.getPosition() - xj.getPosition();
			//}

			if (length(r) > 2 * radius) {
				return 0;
			}

			float outsideTerm = 1 / (M_PI * powf(radius, 5.0f));
			float insideTerm;
			if (length(r) > radius) {
				insideTerm = 1.5 * powf(2 - (length(r) / radius), 1.0f);
			}
			else {
				insideTerm = 3.0f * (-1.0f + 1.5f * (length(r) / radius));
			}

			// otherwise
			return outsideTerm * insideTerm;
		}
};

#endif