#include "Genetic.h"
#include "StandardOffspringMaker.h"
#include "ParallelOffspringMaker.h"
#include <chrono>
#include <fstream>
#include <iomanip>

void Genetic::run()
{
	using Clock = std::chrono::steady_clock;
	auto wallStart = Clock::now();
	auto lastSample = wallStart;

	/* CONVERGENCE TRACE FILE */
	std::ofstream convOut;
	if (!params.convergenceCsvPath.empty())
	{
		convOut.open(params.convergenceCsvPath);
		if (convOut.is_open())
			convOut << "elapsed_sec;best_cost;iter\n";
	}

	auto writeSample = [&](int iter)
	{
		if (!convOut.is_open()) return;
		double elapsed = std::chrono::duration<double>(Clock::now() - wallStart).count();
		const Individual* best = population.getBestFound();
		convOut << std::fixed << std::setprecision(3) << elapsed << ";";
		if (best)
			convOut << std::setprecision(2) << best->eval.penalizedCost;
		else
			convOut << "inf";
		convOut << ";" << iter << "\n";
		convOut.flush();
		lastSample = Clock::now();
	};

	/* INITIAL POPULATION */
	population.generatePopulation();
	writeSample(0);

	int nbIter;
	int nbIterNonProd = 1;
	if (params.verbose) std::cout << "----- STARTING GENETIC ALGORITHM" << std::endl;
	for (nbIter = 0 ; nbIterNonProd <= params.ap.nbIter && (params.ap.timeLimit == 0 || (double)(clock()-params.startTime)/(double)CLOCKS_PER_SEC < params.ap.timeLimit) ; nbIter++)
	{
		bool isNewBest = offspringMaker->makeOffspring();

		/* TRACKING THE NUMBER OF ITERATIONS SINCE LAST SOLUTION IMPROVEMENT */
		if (isNewBest) nbIterNonProd = 1;
		else nbIterNonProd++;

		/* DIVERSIFICATION, PENALTY MANAGEMENT AND TRACES */
		if (nbIter % params.ap.nbIterPenaltyManagement == 0) population.managePenalties();
		if (nbIter % params.ap.nbIterTraces == 0) population.printState(nbIter, nbIterNonProd);

		/* FOR TESTS INVOLVING SUCCESSIVE RUNS UNTIL A TIME LIMIT: WE RESET THE ALGORITHM/POPULATION EACH TIME maxIterNonProd IS ATTAINED*/
		if (params.ap.timeLimit != 0 && nbIterNonProd == params.ap.nbIter)
		{
			population.restart();
			nbIterNonProd = 1;
		}

		/* MIGRATIONS */
		if (islandModel != nullptr)
		{
			double diversity = islandModel->migrationPolicy->needsDiversity() ? population.getDiversity(population.getFeasibleSubpop()) : -1.0;
			islandModel->updateState(nbIter, nbIterNonProd, isNewBest, params.ap.nbIter, diversity);
			islandModel->handleMigrations(population, split, localSearch, params);
		}

		/* CONVERGENCE SAMPLE — at most once per wall-clock second */
		if (convOut.is_open())
		{
			if (std::chrono::duration<double>(Clock::now() - lastSample).count() >= 1.0)
				writeSample(nbIter);
		}
	}

	writeSample(nbIter);  // final data point

	params.telemetry.totalIterations.store(nbIter, std::memory_order_relaxed);
	if (params.verbose) std::cout << "----- GENETIC ALGORITHM FINISHED AFTER " << nbIter << " ITERATIONS. TIME SPENT: " << (double)(clock() - params.startTime) / (double)CLOCKS_PER_SEC << std::endl;
}

void Genetic::crossoverOX(Individual & result, const Individual & parent1, const Individual & parent2)
{
	// Frequency table to track the customers which have been already inserted
	std::vector <bool> freqClient = std::vector <bool> (params.nbClients + 1, false);

	// Picking the beginning and end of the crossover zone
	std::uniform_int_distribution<> distr(0, params.nbClients-1);
	int start = distr(params.ran);
	int end = distr(params.ran);

	// Avoid that start and end coincide by accident
	while (end == start) end = distr(params.ran);

	// Copy from start to end
	int j = start;
	while (j % params.nbClients != (end + 1) % params.nbClients)
	{
		result.chromT[j % params.nbClients] = parent1.chromT[j % params.nbClients];
		freqClient[result.chromT[j % params.nbClients]] = true;
		j++;
	}

	// Fill the remaining elements in the order given by the second parent
	for (int i = 1; i <= params.nbClients; i++)
	{
		int temp = parent2.chromT[(end + i) % params.nbClients];
		if (freqClient[temp] == false)
		{
			result.chromT[j % params.nbClients] = temp;
			j++;
		}
	}

	// Complete the individual with the Split algorithm
	split.generalSplit(result, parent1.eval.nbRoutes);
}

Genetic::Genetic(Params& params)
	: Genetic(params, static_cast<IslandModel*>(nullptr))
{
}

Genetic::Genetic(Params& params, IslandModel& islandModel_)
	: Genetic(params, &islandModel_)
{
}

Genetic::Genetic(Params& params, IslandModel* islandModel_)
	: params(params)
	, split(params)
	, localSearch(params)
	, population(params, this->split, this->localSearch)
	, offspring(params)
	, islandModel(islandModel_)
{
	switch (params.ap.makeManyOffspring)
	{
	case 0:
		offspringMaker = std::make_unique<StandardOffspringMaker>(params, population, localSearch, split);
		break;
	default:
		offspringMaker = std::make_unique<ParallelOffspringMaker>(params, population, params.ap.numOffspring, params.ap.numThreadsOffspring);
		break;
	}
}