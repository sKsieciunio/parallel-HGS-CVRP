#ifndef STANDARD_OFFSPRING_MAKER_H
#define STANDARD_OFFSPRING_MAKER_H

#include "OffspringMaker.h"

class StandardOffspringMaker : public OffspringMaker
{
  public:
	Params & params;
	Population & population;
	LocalSearch & localSearch;
	Split & split;
	Individual offspring;
	std::vector<bool> freqClient;
	std::uniform_int_distribution<> distr;

	void crossoverOX(Individual & result, const Individual & parent1, const Individual & parent2);

	StandardOffspringMaker(Params & params, Population & population, LocalSearch & localSearch, Split & split)
		: params(params), population(population), localSearch(localSearch), split(split), offspring(params), freqClient(params.nbClients + 1, false), distr(0, params.nbClients - 1)
	{}

	bool makeOffspring() override;
};

#endif