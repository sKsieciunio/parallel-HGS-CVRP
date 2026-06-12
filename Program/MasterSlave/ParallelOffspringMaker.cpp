#include "ParallelOffspringMaker.h"
#include <omp.h>
#include <chrono>

ParallelOffspringMaker::ParallelOffspringMaker(Params & params, Population & population, int numOffspring, int numThreads)
	: params(params), population(population),
	  numOffspring(numOffspring), numThreads(numThreads)
{
	localSearches.reserve(numThreads);
	splits.reserve(numThreads);
	rngs.reserve(numOffspring);
	offspring.reserve(numOffspring);
	for (int i = 0; i < numOffspring; i++)
	{
		rngs.emplace_back(params.ap.seed + i);
		offspring.emplace_back(params);
	}

	if (params.ap.useGpu) 
	{
		const int n = params.nbClients + 1;
		std::vector<double> tc((size_t)n * n), dem(n), svc(n);
		for (int i = 0; i < n; i++) 
		{
			dem[i] = params.cli[i].demand;
			svc[i] = params.cli[i].serviceDuration;
			for (int j = 0; j < n; j++)
				tc[(size_t)i * n + j] = params.timeCost[i][j];
		}
		sharedGpu_ = createGpuProblemData(tc.data(), dem.data(), svc.data(), params.nbClients);
	}

	for (int t = 0; t < numThreads; t++) {
		localSearches.push_back(std::make_unique<LocalSearch>(params, std::minstd_rand(params.ap.seed + t), sharedGpu_));
		splits.push_back(std::make_unique<Split>(params));
	}


	needsRepair.resize(numOffspring, false);
	offspringFreqClient.resize(numOffspring * (params.nbClients + 1), false);

	omp_set_num_threads(numThreads);
}

void ParallelOffspringMaker::crossoverOX(Individual & result, const Individual & parent1, const Individual & parent2, std::minstd_rand & rng, int offspringIdx)
{
	int offset = offspringIdx * (params.nbClients + 1);
	std::fill(offspringFreqClient.begin() + offset, offspringFreqClient.begin() + offset + params.nbClients + 1, false);

	std::uniform_int_distribution<> distr(0, params.nbClients - 1);
	int start = distr(rng);
	int end = distr(rng);
	while (end == start)
		end = distr(rng);

	int j = start;
	while (j % params.nbClients != (end + 1) % params.nbClients)
	{
		result.chromT[j % params.nbClients] = parent1.chromT[j % params.nbClients];
		offspringFreqClient[offset + result.chromT[j % params.nbClients]] = true;
		j++;
	}
	for (int i = 1; i <= params.nbClients; i++)
	{
		int temp = parent2.chromT[(end + i) % params.nbClients];
		if (!offspringFreqClient[offset + temp])
		{
			result.chromT[j % params.nbClients] = temp;
			j++;
		}
	}
}

bool ParallelOffspringMaker::makeOffspring()
{
	auto tStart = std::chrono::steady_clock::now();

	for (int i = 1; i <= params.nbClients; i++)
		if (rngs[0]() % params.ap.nbGranular == 0)
			std::shuffle(params.correlatedVertices[i].begin(), params.correlatedVertices[i].end(), rngs[0]);

	population.updateAllBiasedFitnesses();

#pragma omp parallel for schedule(dynamic) num_threads(numThreads)
	for (int i = 0; i < numOffspring; i++) 
	{
		int tid = omp_get_thread_num();
		const Individual & p1 = population.getBinaryTournamentNoUpdate(rngs[i]);
		const Individual & p2 = population.getBinaryTournamentNoUpdate(rngs[i]);
		crossoverOX(offspring[i], p1, p2, rngs[i], i);
		splits[tid]->generalSplit(offspring[i], params.nbVehicles);
		localSearches[tid]->run(offspring[i], params.penaltyCapacity, params.penaltyDuration);
	}

	bool anyNewBest = false;
	std::fill(needsRepair.begin(), needsRepair.end(), false);
	for (int i = 0; i < numOffspring; i++) 
	{
		bool isNewBest = population.addIndividual(offspring[i], true);
		if (isNewBest) anyNewBest = true;
		if (!offspring[i].eval.isFeasible && rngs[i]() % 2 == 0)
			needsRepair[i] = true;
	}

#pragma omp parallel for schedule(dynamic) num_threads(numThreads)
	for (int i = 0; i < numOffspring; i++)
	{
		if (!needsRepair[i]) continue;
		int tid = omp_get_thread_num();
		localSearches[tid]->run(offspring[i], params.penaltyCapacity * 10., params.penaltyDuration * 10.);
	}

	for (int i = 0; i < numOffspring; i++)
	{
		if (!needsRepair[i]) continue;
		if (offspring[i].eval.isFeasible) 
		{
			if (population.addIndividual(offspring[i], false))
				anyNewBest = true;
		}
	}

	params.telemetry.offspringRounds.fetch_add(1, std::memory_order_relaxed);
	params.telemetry.offspringGenerated.fetch_add(numOffspring, std::memory_order_relaxed);
	params.telemetry.offspringNs.fetch_add(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(
	        std::chrono::steady_clock::now() - tStart).count(),
	    std::memory_order_relaxed);

	return anyNewBest;
}

ParallelOffspringMaker::~ParallelOffspringMaker()
{
	destroyGpuProblemData(sharedGpu_);
}