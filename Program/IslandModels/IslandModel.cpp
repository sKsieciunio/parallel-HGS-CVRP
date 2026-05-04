#include "IslandModel.h"
#include "../IslandCommunicators/MPIIslandCommunicator.h"

IslandModel::IslandModel(Params& params)
{
    switch (ap.islandCommunicator) {
    case 0:
    default:
#ifdef USE_MPI
        islandCommunicator = std::make_unique<MPIIslandCommunicator>(params);
#else
        throw std::string("MPI communicator requested but MPI not available.");
#endif
        break;
    }

    int rank = islandCommunicator->getRank();
    int nIslands = ap.nbNodes;

    switch (ap.topology) {
    case 0: topology = std::make_unique<RingTopology>(nIslands, rank); break;
    case 1: topology = std::make_unique<TwoSidedRing>(nIslands, rank); break;
    case 2: topology = std::make_unique<Hypercube>(nIslands, rank); break;
    case 3: topology = std::make_unique<StarTopology>(nIslands, rank); break;
    case 4: topology = std::make_unique<FullGraphTopology>(nIslands, rank); break;
    default: topology = std::make_unique<RingTopology>(nIslands, rank); break;
    }

    switch (ap.migrationPolicy) {
    case 0:
        migrationPolicy = std::make_unique<FixedIntervalMigrationPolicy>(
            ap.interval);
        break;
    case 1:
        migrationPolicy = std::make_unique<ImprovementTriggeredMigrationPolicy>(
            ap.warmup, ap.sendCooldown, ap.receiveStagnationThreshold);
        break;
    case 2:
        migrationPolicy = std::make_unique<AdaptiveMigrationPolicy>(
            ap.sendCooldown, ap.minReceiveInterval,
            ap.maxReceiveInterval, ap.warmup);
        break;
    default:
        migrationPolicy = std::make_unique<FixedIntervalMigrationPolicy>(
            ap.interval);
        break;
    }

    switch (ap.migrantSelector) {
    case 0:
        migrantSelector = std::make_unique<StandardMigrantSelector>(
            ap.selectionCount);
        break;
    default:
        migrantSelector = std::make_unique<StandardMigrantSelector>(
            ap.selectionCount);
        break;
    }

    switch (ap.immigrantHandler) {
    case 0: immigrantHandler = std::make_unique<StandardImmigrantHandler>(); break;
    case 1: immigrantHandler = std::make_unique<LocalSearchImmigrantHandler>(); break;
    case 2: immigrantHandler = std::make_unique<RepairImmigrantHandler>(); break;
    default: immigrantHandler = std::make_unique<StandardImmigrantHandler>(); break;
    }

    islandState = { 0, 0, false, ap.nbIter };
}

void IslandModel::updateState(int iteration, int iterWithoutImprovement,
    bool foundNewBest, int maxIterNoImprovement)
{
    islandState.iteration = iteration;
    islandState.iterationWithoutImprovement = iterWithoutImprovement;
    islandState.foundNewBest = foundNewBest;
    islandState.maxIterNoImprovement = maxIterNoImprovement;
}

void IslandModel::handleMigrations(Population& population, Split& split,
    LocalSearch& localSearch, Params& params)
{
    if (migrationPolicy->shouldSend(islandState)) {
        islandCommunicator.sendMigrants(
            migrantSelector->selectMigrants(population), topology->getNeighbors());
    }

    if (migrationPolicy->shouldReceive(islandState)) {
        auto incoming = islandCommunicator.tryReceiveMigrants();
        for (auto& migrant : incoming) {
            immigrantHandler->handle(population, migrant, split, localSearch, params);
        }
    }
}