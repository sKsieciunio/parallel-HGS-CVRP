#ifndef PARALLEL_OFFSPRING_MAKER_H
#define PARALLEL_OFFSPRING_MAKER_H

#include "OffspringMaker.h"

class ParallelOffspringMaker : public OffspringMaker
{
  public:
	int numThreads;
	std::vector<LocalSearch> localSearches;
	std::vector<Split> splits;
	std::vector<std::minstd_rand> rngs;
	std::vector<Individual> offspring;
	std::vector<bool> needsRepair;
	Population * population;

	void crossoverOX(Individual & result, const Individual & parent1, const Individual & parent2, int threadIdx);
	bool makeOffspring() override;

	ParallelOffspringMaker(Params & params, Population * population);
	~ParallelOffspringMaker();
};

#endif // !OFFSPRING_MAKER_H
