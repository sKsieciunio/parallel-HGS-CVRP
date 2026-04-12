#ifndef MPI_ISLAND_COMMUNICATOR_H
#define MPI_ISLAND_COMMUNICATOR_H

#include "IslandCommunicator.h"
#define USE_MPI
#ifdef USE_MPI
#include <mpi.h>

class MPIIslandCommunicator : public IslandCommunicator {
private:
    Params& params;
    int rank;
    int size;

    std::vector<int> recvBuffer;
    MPI_Request recvRequest;

    struct PendingSend {
        std::vector<int> buffer;
        MPI_Request request;
    };
    std::list<PendingSend> pendingSends;

    bool doneSignaled = false;
    bool stopFlag = false;
    int doneBuffer = 0;
    MPI_Request doneRecvRequest;

    static const int TAG_MIGRANT = 0;
    static const int TAG_DONE = 1;

    void cleanupPendingSends();

    void postReceive();

public:
    MPIIslandCommunicator(Params& params);

    void sendMigrants(const std::vector<Individual*>& migrants, const std::vector<int>& destinations) override;

    std::vector<Individual> tryReceiveMigrants() override;

    void signalDone() override;

    bool shouldStop() override;

    int getRank() const { return rank; }
    int getSize() const { return size; }

    ~MPIIslandCommunicator();
};

#endif // USE_MPI
#endif // !MPI_ISLAND_COMMUNICATOR_H
