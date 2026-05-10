#ifndef SYNC_MPI_ISLAND_COMMUNICATOR_H
#define SYNC_MPI_ISLAND_COMMUNICATOR_H

#include "IslandCommunicator.h"

#ifdef USE_MPI
#include <mpi.h>

class SynchronousMPIIslandCommunicator : public IslandCommunicator {
private:
    Params& params;
    int rank;
    int size;

    std::vector<Individual> receivedBuffer;
    std::vector<int> recvBuf;
    std::vector<MPI_Request> sendRequests;
    std::vector<std::vector<int>> sendBuffers;

    static const int TAG_MIGRANT = 0;

public:
    SynchronousMPIIslandCommunicator(Params& params) : params(params) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        recvBuf.resize(params.nbClients);
    }

    void sendMigrants(const std::vector<Individual*>& migrants, const std::vector<int>& destinations) override;
    std::vector<Individual> tryReceiveMigrants() override;

    int getRank() const override;
    int getSize() const;

    bool getBestSolution(const Individual* bestLocal, int nbClients, std::vector<int>& outBestChromT, double& outBestCost) override;

    ~SynchronousMPIIslandCommunicator() = default;
};


#endif // USE_MPI
#endif // !SYNC_MPI_ISLAND_COMMUNICATOR_H