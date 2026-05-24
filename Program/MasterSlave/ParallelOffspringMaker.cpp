#include "ParallelOffspringMaker.h"
#include <omp.h>

ParallelOffspringMaker::ParallelOffspringMaker(Params & params, Population * population, int numOffspring, int numThreads)
	: params(params), population(population),
	  numOffspring(numOffspring), numThreads(numThreads)
{
	localSearches.reserve(numThreads);
	splits.reserve(numThreads);
	for (int t = 0; t < numThreads; t++) {
		localSearches.emplace_back(params);
		splits.emplace_back(params);
	}

	rngs.reserve(numOffspring);
	offspring.reserve(numOffspring);
	for (int i = 0; i < numOffspring; i++) {
		rngs.emplace_back(params.ap.seed + i * 77 + 13);
		offspring.emplace_back(params);
	}

	needsRepair.resize(numOffspring, false);
	distr = std::uniform_int_distribution<>(0, params.nbClients - 1);
	offspringFreqClient.resize(numOffspring * (params.nbClients + 1), false);

	omp_set_num_threads(numThreads);
}

void ParallelOffspringMaker::crossoverOX(Individual & result, const Individual & parent1, const Individual & parent2, std::minstd_rand & rng, int offspringIdx)
{
	int offset = offspringIdx * (params.nbClients + 1);
	std::fill(offspringFreqClient.begin() + offset, offspringFreqClient.begin() + offset + params.nbClients + 1, false);

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
	population->updateAllBiasedFitnesses();

#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < numOffspring; i++) 
	{
		int tid = omp_get_thread_num();
		const Individual & p1 = population->getBinaryTournamentNoUpdate(rngs[i]);
		const Individual & p2 = population->getBinaryTournamentNoUpdate(rngs[i]);
		crossoverOX(offspring[i], p1, p2, rngs[i], i);
		splits[tid].generalSplit(offspring[i], params.nbVehicles);
		localSearches[tid].run(offspring[i], params.penaltyCapacity, params.penaltyDuration);
	}

	bool anyNewBest = false;
	std::fill(needsRepair.begin(), needsRepair.end(), false);
	for (int i = 0; i < numOffspring; i++) 
	{
		bool isNewBest = population->addIndividual(offspring[i], true);
		if (isNewBest) anyNewBest = true;
		if (!offspring[i].eval.isFeasible && rngs[i]() % 2 == 0)
			needsRepair[i] = true;
	}

#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < numOffspring; i++)
	{
		if (!needsRepair[i]) continue;
		int tid = omp_get_thread_num();
		localSearches[tid].run(offspring[i], params.penaltyCapacity * 10., params.penaltyDuration * 10.);
	}

	for (int i = 0; i < numOffspring; i++)
	{
		if (!needsRepair[i]) continue;
		if (offspring[i].eval.isFeasible) 
		{
			if (population->addIndividual(offspring[i], false))
				anyNewBest = true;
		}
	}

	return anyNewBest;
}