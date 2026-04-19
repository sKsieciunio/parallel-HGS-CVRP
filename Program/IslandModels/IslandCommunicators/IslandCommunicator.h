#ifndef ISLAND_COMMUNICATOR_H
#define ISLAND_COMMUNICATOR_H

#include "../../Individual.h"

class IslandCommunicator {
public:
    virtual ~IslandCommunicator() = default;
    virtual void sendMigrants(const std::vector<Individual*>& migrants, const std::vector<int>& destinations) = 0;
    virtual std::vector<Individual> tryReceiveMigrants() = 0;
    virtual int getRank() const = 0;
};

#endif // !ISLAND_COMMUNICATOR_H
