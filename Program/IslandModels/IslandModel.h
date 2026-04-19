#ifndef ISLAND_MODEL_H
#define ISLAND_MODEL_H

#include "../Topologies/Topology.h"
#include "../ImmigrantHandlers/ImmigrantHandler.h"
#include "../IslandCommunicators/IslandCommunicator.h"
#include "../MigrationPolicies/MigrationPolicy.h"
#include "../MigrantSelectors/MigrantSelector.h"

class IslandModel {
private:
	Topology& topology;
	ImmigrantHandler& immigrantHandler;
	IslandCommunicator& islandCommunicator;
	MigrationPolicy& migrationPolicy;
	MigrantSelector& migrantSelector;
	IslandState islandState;

public:
	IslandModel(Topology& topology_, ImmigrantHandler& immigrantHandler_, IslandCommunicator& islandCommunicator_, MigrationPolicy& migrationPolicy_, MigrantSelector& migrantSelector_, IslandState& islandState_) :
		topology(topology_), immigrantHandler(immigrantHandler_), islandCommunicator(islandCommunicator_), migrationPolicy(migrationPolicy_), migrantSelector(migrantSelector_), islandState(islandState_)
	{ }

	void UpdateState(int iteration, int iterationWithoutImprovement, bool foundNewBest, int maxIterNoImprovement);

	void HandleMigrations(Population& population, const Split& split, const LocalSearch& localSearch, const Params& params);
};

#endif // !ISLAND_MODEL_H
