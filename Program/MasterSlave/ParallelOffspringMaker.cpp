#include "ParallelOffspringMaker.h"
#include <omp.h>

ParallelOffspringMaker::ParallelOffspringMaker(Params & params, Population * population)
	: params(params), population(population)
{
	numThreads = omp_get_max_threads();
	for (int t = 0; t < numThreads; t++) {
		localSearches.emplace_back(params);
		splits.emplace_back(params);
		rngs.emplace_back(params.ap.seed + t * 77 + 13);
		offspring.emplace_back(params);
	}
	needsRepair.resize(numThreads, false);
}

void ParallelOffspringMaker::crossoverOX(Individual & result, const Individual & parent1, const Individual & parent2, std::minstd_rand & rng)
{
	std::vector<bool> freqClient(params.nbClients + 1, false);
	std::uniform_int_distribution<> distr(0, params.nbClients - 1);
	int start = distr(rng);
	int end = distr(rng);
	while (end == start)
		end = distr(rng);

	int j = start;
	while (j % params.nbClients != (end + 1) % params.nbClients) {
		result.chromT[j % params.nbClients] = parent1.chromT[j % params.nbClients];
		freqClient[result.chromT[j % params.nbClients]] = true;
		j++;
	}
	for (int i = 1; i <= params.nbClients; i++) {
		int temp = parent2.chromT[(end + i) % params.nbClients];
		if (!freqClient[temp]) {
			result.chromT[j % params.nbClients] = temp;
			j++;
		}
	}
}

bool ParallelOffspringMaker::makeOffspring()
{
	population->updateAllBiasedFitnesses();

#pragma omp parallel
	{
		int tid = omp_get_thread_num();

		const Individual & p1 = population->getBinaryTournamentNoUpdate(rngs[tid]);
		const Individual & p2 = population->getBinaryTournamentNoUpdate(rngs[tid]);

		crossoverOX(offspring[tid], p1, p2, rngs[tid]);
		splits[tid].generalSplit(offspring[tid], params.nbVehicles);
		localSearches[tid].run(offspring[tid], params.penaltyCapacity, params.penaltyDuration);
	}

	bool anyNewBest = false;
	std::fill(needsRepair.begin(), needsRepair.end(), false);

	for (int t = 0; t < numThreads; t++) {
		bool isNewBest = population->addIndividual(offspring[t], true);
		if (isNewBest) anyNewBest = true;

		if (!offspring[t].eval.isFeasible && rngs[t]() % 2 == 0)
			needsRepair[t] = true;
	}

#pragma omp parallel for
	for (int t = 0; t < numThreads; t++) 
	{
		if (!needsRepair[t]) continue;
		localSearches[t].run(offspring[t], params.penaltyCapacity * 10., params.penaltyDuration * 10.);
	}

	for (int t = 0; t < numThreads; t++) 
	{
		if (!needsRepair[t]) continue;
		if (offspring[t].eval.isFeasible) 
		{
			if (population->addIndividual(offspring[t], false))
				anyNewBest = true;
		}
	}

	return anyNewBest;
}