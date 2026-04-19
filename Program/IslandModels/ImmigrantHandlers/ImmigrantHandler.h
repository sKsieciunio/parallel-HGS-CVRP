#ifndef ACCAPTANCE_POLICY_H
#define ACCEPTANCE_POLICY_H

#include "../../Population.h"

class ImmigrantHandler {
public:
	virtual ~ImmigrantHandler() = default;
	virtual void handle(Population& population, Individual& migrant, const Split& split, const LocalSearch& localSearch, const Params& params) const = 0;
};

class StandardImmigrantHandler : public ImmigrantHandler {
public:
	void handle(Population& population, Individual& migrant, const Split& split, const LocalSearch& localSearch, const Params& params) override const {
		split.generalSplit(migrant, params.nbVehicles);
		population.addIndividual(migrant, false);
	}
};

class LocalSearchImmigrantHandler : public ImmigrantHandler {
public:
	void handle(Population& population, Individual& migrant, const Split& split, const LocalSearch& localSearch, const Params& params) override const {
		split.generalSplit(migrant, params.nbVehicles);
		localSearch.run(migrant, params.penaltyCapacity, params.penaltyDuration);
		population.addIndividual(migrant, false);
	}
};

class RepairImmigrantHandler : public ImmigrantHandler {
public:
	void handle(Population& population, Individual& migrant, const Split& split, const LocalSearch& localSearch, const Params& params) override const {
		split.generalSplit(migrant, params.nbVehicles);
		localSearch.run(migrant, params.penaltyCapacity, params.penaltyDuration);
		if (!migrant.eval.isFeasible) {
			localSearch.run(migrant, params.penaltyCapacity * 10., params.penaltyDuration * 10.);
		}
		population.addIndividual(migrant, false);
	}
};

#endif // !ACCAPTANCE_POLICY_H
