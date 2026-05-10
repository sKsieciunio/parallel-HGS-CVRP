#include "IslandModel.h"
#include "../IslandCommunicators/MPIIslandCommunicator.h"
#include "../IslandCommunicators/SynchronousMPIIslandCommunicator.h"

IslandModel::IslandModel(Params& params)
{
    switch (params.ap.islandCommunicator) {
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
    int nIslands = params.ap.nbNodes;

    switch (params.ap.topology) {
    case 0: topology = std::make_unique<RingTopology>(nIslands, rank); break;
    case 1: topology = std::make_unique<TwoSidedRing>(nIslands, rank); break;
    case 2: topology = std::make_unique<Hypercube>(nIslands, rank); break;
    case 3: topology = std::make_unique<StarTopology>(nIslands, rank); break;
    case 4: topology = std::make_unique<FullGraphTopology>(nIslands, rank); break;
    default: topology = std::make_unique<RingTopology>(nIslands, rank); break;
    }

    switch (params.ap.migrationPolicy) {
    case 0:
        migrationPolicy = std::make_unique<FixedIntervalMigrationPolicy>(
            params.ap.interval);
        break;
    case 1:
        migrationPolicy = std::make_unique<ImprovementTriggeredMigrationPolicy>(
            params.ap.warmup, params.ap.sendCooldown, params.ap.receiveStagnationThreshold);
        break;
    case 2:
        migrationPolicy = std::make_unique<AdaptiveMigrationPolicy>(
            params.ap.sendCooldown, params.ap.minReceiveInterval,
            params.ap.maxReceiveInterval, params.ap.warmup);
        break;
    default:
        migrationPolicy = std::make_unique<FixedIntervalMigrationPolicy>(
            params.ap.interval);
        break;
    }

    switch (params.ap.migrantSelector) {
    case 0:
        migrantSelector = std::make_unique<StandardMigrantSelector>(
            params.ap.selectionCount);
        break;
    default:
        migrantSelector = std::make_unique<StandardMigrantSelector>(
            params.ap.selectionCount);
        break;
    }

    switch (params.ap.immigrantHandler) {
    case 0: immigrantHandler = std::make_unique<StandardImmigrantHandler>(); break;
    case 1: immigrantHandler = std::make_unique<LocalSearchImmigrantHandler>(); break;
    case 2: immigrantHandler = std::make_unique<RepairImmigrantHandler>(); break;
    default: immigrantHandler = std::make_unique<StandardImmigrantHandler>(); break;
    }

    islandState = { 0, 0, false, params.ap.nbIter };
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
        islandCommunicator->sendMigrants(
            migrantSelector->selectMigrants(population), topology->getNeighbors());
    }

    if (migrationPolicy->shouldReceive(islandState)) {
        auto incoming = islandCommunicator->tryReceiveMigrants();
        for (auto& migrant : incoming) {
            immigrantHandler->handle(population, migrant, split, localSearch, params);
        }
    }
}

bool IslandModel::getBestSolution(Population& population, Split& split, Params& params, Individual& bestOut) 
{
    const Individual* localBest = population.getBestFound();
    std::vector<int> bestChromT;
    double bestCost;

    bool shouldExport = islandCommunicator->getBestSolution(localBest, params.nbClients, bestChromT, bestCost);

    if (shouldExport) {
        if (localBest && localBest->eval.penalizedCost <= bestCost + MY_EPSILON) {
            bestOut = *localBest;
        }
        else {
            bestOut = Individual(params, bestChromT);
            split.generalSplit(bestOut, params.nbVehicles);
        }
    }
    return shouldExport;
}