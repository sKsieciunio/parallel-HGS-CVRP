#include "StandardOffspringMaker.h"

bool StandardOffspringMaker::makeOffspring() 
{
	const Individual & p1 = population.getBinaryTournament();
	const Individual & p2 = population.getBinaryTournament();

	crossoverOX(offspring, p1, p2);
	localSearch.run(offspring, params.penaltyCapacity, params.penaltyDuration);

	bool isNewBest = population.addIndividual(offspring, true);

	if (!offspring.eval.isFeasible && params.ran() % 2 == 0) {
		localSearch.run(offspring, params.penaltyCapacity * 10.,
						params.penaltyDuration * 10.);
		if (offspring.eval.isFeasible)
			isNewBest = population.addIndividual(offspring, false) || isNewBest;
	}

	return isNewBest;
}

void StandardOffspringMaker::crossoverOX(Individual& result, const Individual& parent1, const Individual& parent2)
{
	std::fill(freqClient.begin(), freqClient.end(), false); 

	// Picking the beginning and end of the crossover zone
	int start = distr(params.ran);
	int end = distr(params.ran);

	// Avoid that start and end coincide by accident
	while (end == start)
		end = distr(params.ran);

	// Copy from start to end
	int j = start;
	while (j % params.nbClients != (end + 1) % params.nbClients) {
		result.chromT[j % params.nbClients] = parent1.chromT[j % params.nbClients];
		freqClient[result.chromT[j % params.nbClients]] = true;
		j++;
	}

	// Fill the remaining elements in the order given by the second parent
	for (int i = 1; i <= params.nbClients; i++) {
		int temp = parent2.chromT[(end + i) % params.nbClients];
		if (freqClient[temp] == false) {
			result.chromT[j % params.nbClients] = temp;
			j++;
		}
	}

	// Complete the individual with the Split algorithm
	split.generalSplit(result, parent1.eval.nbRoutes);
}