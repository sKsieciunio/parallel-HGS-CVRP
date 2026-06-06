#ifndef ISLAND_STATE_H
#define ISLAND_STATE_H

struct IslandState
{
	int iteration;
	int iterationWithoutImprovement;
	bool foundNewBest;
	int maxIterNoImprovement;
	double diversity = -1.0;
};

#endif // !ISLAND_STATE_H
