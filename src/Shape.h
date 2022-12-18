#pragma once
#ifndef SHAPE_H
#define SHAPE_H

#include <string>
#include <vector>
#include <memory>

class Program;

#define EIGEN_DONT_ALIGN_STATICALLY
#include <Eigen/Eigen>

class Shape
{
public:
	Shape();
	virtual ~Shape();
	void loadMesh(const std::string &meshName);
	void init();
	void draw(const std::shared_ptr<Program> prog) const;
	std::vector<Eigen::Matrix<float, 3, 1>> sampleMesh(float particleRadius);
	
private:
	std::vector<float> posBuf;
	std::vector<float> norBuf;
	std::vector<float> texBuf;
	unsigned posBufID;
	unsigned norBufID;
	unsigned texBufID;
	Eigen::Matrix<unsigned int, 3, -1> faceBuf;
	Eigen::Matrix<float, 3, -1> posBuf_eigen;
};

#endif
