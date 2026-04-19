#include "IslandModel.h"

void IslandModel::UpdateState(int iteration, int iterationWithoutImprovement, bool foundNewBest, int maxIterNoImprovement)
{
	islandState.iteration = iteration;
	islandState.iterationWithoutImprovement = iterationWithoutImprovement;
	islandState.foundNewBest = foundNewBest;
	islandState.maxIterNoImprovement = maxIterNoImprovement;
}

void IslandModel::HandleMigrations(Population& population, const Split& split, const LocalSearch& localSearch, const Params& params)
{
	if (migrationPolicy.shouldSend(islandState)) {
		islandCommunicator.sendMigrants(migrantSelector.selectMigrants(population), topology.getNeighbors());
	}

	if (migrationPolicy.shouldReceive(islandState)) {
		for each(Individual& im in islandCommunicator.tryReceiveMigrants()) {
			immigrantHandler.handle(population, im, split, localSearch, params);
		}
	}
}
