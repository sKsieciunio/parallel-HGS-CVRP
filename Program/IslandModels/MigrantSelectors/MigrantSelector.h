#ifndef MIGRANT_SELECTOR_H
#define MIGRANT_SELECTOR_H

#include "../../Population.h"

class MigrantSelector {
public:
    virtual ~MigrantSelector() = default;
    virtual std::vector<Individual*> selectMigrants(const Population& population) const = 0;
};

class StandardMigrantSelector : public MigrantSelector {
private:
    int selectionCount;
public:
    StandardMigrantSelector(int selectionCount) : selectionCount(selectionCount) {}

    std::vector<Individual*> selectMigrants(const Population& population) const override {
        const auto& subpop = population.getFeasibleSubpop();
        int n = std::min(selectionCount, (int)subpop.size());
        return std::vector<Individual*>(subpop.begin(), subpop.begin() + n);
    }
};

#endif