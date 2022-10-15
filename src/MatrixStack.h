#pragma once
#ifndef _MATRIX_STACK_H_
#define _MATRIX_STACK_H_

#include <stack>
#include <memory>
#include <glm/fwd.hpp>

class MatrixStack
{
public:

	// initializes as an identity matrix
	MatrixStack();
	virtual ~MatrixStack();
	
	// pushes and pops
	void pushMatrix();
	void popMatrix();

	// manually multiplying matrix- to the right!
	void multMatrix(const glm::mat4&);

	// automatic transforms
	void translate(const glm::vec3& translation);
	void scale(const glm::vec3& scaleFactor);
	void rotate(float angle, const glm::vec3& rotationAxis);

	//get the top matrix
	glm::mat4& MatrixStack::topMatrix() const;
	
private:
	std::shared_ptr<std::stack<glm::mat4>> matrixStack;
	
};

#endif
