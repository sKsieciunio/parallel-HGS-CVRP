#ifndef PARALLEL_OFFSPRING_MAKER_H
#define PARALLEL_OFFSPRING_MAKER_H
#include "OffspringMaker.h"
#include <memory>

class ParallelOffspringMaker : public OffspringMaker
{
  public:
	Params & params;
	Population & population;
	int numOffspring;
	int numThreads;

	std::vector<std::unique_ptr<LocalSearch>> localSearches;
	std::vector < std::unique_ptr<Split>> splits;
	std::vector<std::minstd_rand> rngs;
	std::vector<Individual> offspring;
	std::vector<bool> needsRepair;
	std::vector<char> offspringFreqClient;
	std::uniform_int_distribution<> distr;

	void crossoverOX(Individual & result, const Individual & parent1, const Individual & parent2, std::minstd_rand & rng, int offspringIdx);

	bool makeOffspring() override;
	ParallelOffspringMaker(Params & params, Population & population, int numOffspring, int numThreads);
	~ParallelOffspringMaker() = default;
};

#endif