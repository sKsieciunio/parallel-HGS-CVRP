#include "IslandModel.h"
#include "../IslandCommunicators/MPIIslandCommunicator.h"
#include "../IslandCommunicators/SynchronousMPIIslandCommunicator.h"

std::unique_ptr<IslandCommunicator> IslandModel::makeCommunicator(Params& params)
{
#ifdef USE_MPI
    switch (params.ap.islandCommunicator)
    {
    case 1:  return std::make_unique<SynchronousMPIIslandCommunicator>(params);
    case 0:
    default: return std::make_unique<MPIIslandCommunicator>(params);
    }
#else
    throw std::string("MPI communicator requested but MPI not available.");
#endif
}

std::unique_ptr<Topology> IslandModel::makeTopology(const Params& params, int rank)
{
    int nIslands = params.ap.nbNodes;
    switch (params.ap.topology)
    {
    case 1:  return std::make_unique<TwoSidedRing>(nIslands, rank);
    case 2:  return std::make_unique<Hypercube>(nIslands, rank);
    case 3:  return std::make_unique<StarTopology>(nIslands, rank);
    case 4:  return std::make_unique<FullGraphTopology>(nIslands, rank);
    case 5:  return std::make_unique<RandomRegularTopology>(nIslands, rank, params.ap.topologyDegree, (unsigned)params.ap.seed);
    case 0:
    default: return std::make_unique<RingTopology>(nIslands, rank);
    }
}

std::unique_ptr<MigrationPolicy> IslandModel::makeMigrationPolicy(const Params& params)
{
    switch (params.ap.migrationPolicy)
    {
    case 1:  return std::make_unique<ImprovementTriggeredMigrationPolicy>(params.ap.warmup, params.ap.sendCooldown, params.ap.receiveStagnationThreshold);
    case 2:  return std::make_unique<AdaptiveMigrationPolicy>(params.ap.sendCooldown, params.ap.minReceiveInterval, params.ap.maxReceiveInterval, params.ap.warmup);
    case 3:  return std::make_unique<DiversityDrivenMigrationPolicy>(params.ap.sendCooldown, params.ap.minReceiveInterval, params.ap.maxReceiveInterval, params.ap.warmup);
    case 0:
    default: return std::make_unique<FixedIntervalMigrationPolicy>(params.ap.interval);
    }
}

std::unique_ptr<MigrantSelector> IslandModel::makeMigrantSelector(const Params& params)
{
    switch (params.ap.migrantSelector)
    {
    case 0:
    default: return std::make_unique<StandardMigrantSelector>(params.ap.selectionCount);
    }
}

std::unique_ptr<ImmigrantHandler> IslandModel::makeImmigrantHandler(const Params& params)
{
    switch (params.ap.immigrantHandler)
    {
    case 1:  return std::make_unique<LocalSearchImmigrantHandler>();
    case 2:  return std::make_unique<RepairImmigrantHandler>();
    case 0:
    default: return std::make_unique<StandardImmigrantHandler>();
    }
}

IslandModel::IslandModel(Params& params)
{
    islandCommunicator = makeCommunicator(params);

    int rank = islandCommunicator->getRank();

    topology = makeTopology(params, rank);
    migrationPolicy = makeMigrationPolicy(params);
    migrantSelector = makeMigrantSelector(params);
    immigrantHandler = makeImmigrantHandler(params);

    islandState = { 0, 0, false, params.ap.nbIter };
}

void IslandModel::updateState(int iteration, int iterWithoutImprovement, bool foundNewBest, int maxIterNoImprovement, double diversity)
{
    islandState.iteration = iteration;
    islandState.iterationWithoutImprovement = iterWithoutImprovement;
    islandState.foundNewBest = foundNewBest;
    islandState.maxIterNoImprovement = maxIterNoImprovement;
    islandState.diversity = diversity;
}

void IslandModel::handleMigrations(Population& population, Split& split, LocalSearch& localSearch, Params& params) 
{
    if (migrationPolicy->shouldSend(islandState)) 
    {
        islandCommunicator->sendMigrants(migrantSelector->selectMigrants(population), topology->getNeighbors());
    }

    if (migrationPolicy->shouldReceive(islandState)) 
    {
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

    if (shouldExport)
    {
        if (localBest && localBest->eval.penalizedCost <= bestCost + MY_EPSILON)
        {
            bestOut = *localBest;
        }
        else
        {
            bestOut = Individual(params, bestChromT);
            split.generalSplit(bestOut, params.nbVehicles);
        }
    }
    return shouldExport;
}