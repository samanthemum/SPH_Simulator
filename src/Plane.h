#pragma once
#ifndef _PLANE_H
#define _PLANE_H


#include <memory>
#include <glm/common.hpp>
#include <glm/fwd.hpp>

class Plane {
	public:
		Plane() {}
		Plane(glm::vec3 norm, glm::vec3 p): normal(norm), point(p) {}
		~Plane() {}

		void setNormal(glm::vec3 normal) { this->normal = normal; };
		void setPoint(glm::vec3 point) { this->point = point; }
		glm::vec3 getNormal() const { return this->normal; }
		glm::vec3 getPoint() const { return this->point; }

	private: 
		glm::vec3 normal = glm::vec3(0.0f, 0.0f, 0.0f);
		glm::vec3 point = glm::vec3(0.0f, 0.0f, 0.0f);
};

#endif