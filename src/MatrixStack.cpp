#include "MatrixStack.h"

#include <stdio.h>
#include <cassert>
#include <vector>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

using namespace std;

MatrixStack::MatrixStack()
{
	matrixStack = make_shared< std::stack<glm::mat4> >();
	matrixStack->push(glm::mat4(1.0));
}

MatrixStack::~MatrixStack()
{
}

void MatrixStack::pushMatrix()
{
	const glm::mat4 &top = matrixStack->top();
	matrixStack->push(top);
	assert(matrixStack->size() < 100);
}

void MatrixStack::popMatrix()
{
	assert(!matrixStack->empty());
	matrixStack->pop();
	// There should always be one matrix left.
	assert(!matrixStack->empty());
}

void MatrixStack::translate(const glm::vec3 &t)
{
	glm::mat4 &top = matrixStack->top();
	top *= glm::translate(glm::mat4(1.0f), t);
}

void MatrixStack::scale(const glm::vec3 &scaleFactor)
{
	glm::mat4 &top = matrixStack->top();
	top *= glm::scale(glm::mat4(1.0f), scaleFactor);
}

void MatrixStack::rotate(float angle, const glm::vec3& rotationAxis) {
	glm::mat4& top = matrixStack->top();
	top *= glm::rotate(glm::mat4(1.0f), angle, rotationAxis);
}

void MatrixStack::multMatrix(const glm::mat4 &matrix)
{
	glm::mat4 &top = matrixStack->top();
	top *= matrix;
}

glm::mat4 &MatrixStack::topMatrix() const
{
	return matrixStack->top();
}
