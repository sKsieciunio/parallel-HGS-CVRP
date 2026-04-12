#ifndef ISLAND_STATE_H
#define ISLAND_STATE_H

#include "../Population.h"

struct IslandState {
	int iteration;
	int iterationWithoutImprovement;
	bool foundNewBest;
	int maxIterNoImprovement;
};

#endif // !ISLAND_STATE_H
