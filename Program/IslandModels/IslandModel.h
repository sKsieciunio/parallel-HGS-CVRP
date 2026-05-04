#ifndef ISLAND_MODEL_H
#define ISLAND_MODEL_H

#include <memory>
#include "../Topologies/Topology.h"
#include "../ImmigrantHandlers/ImmigrantHandler.h"
#include "../IslandCommunicators/IslandCommunicator.h"
#include "../MigrationPolicies/MigrationPolicy.h"
#include "../MigrantSelectors/MigrantSelector.h"

class IslandModel {
private:
    std::unique_ptr<Topology> topology;
    std::unique_ptr<ImmigrantHandler> immigrantHandler;
    std::unique_ptr<MigrationPolicy> migrationPolicy;
    std::unique_ptr<MigrantSelector> migrantSelector;
    std::unique_ptr<IslandCommunicator> islandCommunicator;
    IslandState islandState;

public:
    IslandModel(Params& params);

    void updateState(int iteration, int iterWithoutImprovement, bool foundNewBest, int maxIterNoImprovement);
    void handleMigrations(Population& population, Split& split, LocalSearch& localSearch, Params& params);
    bool getBestSolution(Population& population, Split& split, Params& params, Individual& bestOut);
};

#endif