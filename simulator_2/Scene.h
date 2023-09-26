#pragma once
#ifndef SCENE_H
#define SCENE_H
#include "Particle.h"

class Scene {
	private:
		Particle* particleList;
		Vec3f* particlePositions;
};

#endif