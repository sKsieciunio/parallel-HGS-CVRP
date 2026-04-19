#include "MPIIslandCommunicator.h"
#ifdef USE_MPI

void MPIIslandCommunicator::cleanupPendingSends() {
    pendingSends.remove_if([](PendingSend& ps) {
        int flag = 0;
        MPI_Test(&ps.request, &flag, MPI_STATUS_IGNORE);
        return flag;
        });
}

void MPIIslandCommunicator::postReceive() {
    MPI_Irecv(recvBuffer.data(), params.nbClients, MPI_INT, MPI_ANY_SOURCE, TAG_MIGRANT, MPI_COMM_WORLD, &recvRequest);
}

MPIIslandCommunicator::MPIIslandCommunicator(Params& params) : params(params) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    recvBuffer.resize(params.nbClients);
    postReceive();
}

void MPIIslandCommunicator::sendMigrants(const std::vector<Individual*>& migrants, const std::vector<int>& destinations) {
    cleanupPendingSends();
    for (const Individual* migrant : migrants) {
        for (int dest : destinations) {
            pendingSends.push_back({ migrant->chromT, MPI_REQUEST_NULL });
            MPI_Isend(pendingSends.back().buffer.data(), params.nbClients, MPI_INT, dest, TAG_MIGRANT, MPI_COMM_WORLD, &pendingSends.back().request);
        }
    }
}

std::vector<Individual> MPIIslandCommunicator::tryReceiveMigrants() {
    std::vector<Individual> received;
    int flag = 0;
    MPI_Test(&recvRequest, &flag, MPI_STATUS_IGNORE);
    while (flag) {
        std::vector<int> chromT(recvBuffer.begin(), recvBuffer.end());
        received.emplace_back(params, chromT);
        postReceive();
        flag = 0;
        MPI_Test(&recvRequest, &flag, MPI_STATUS_IGNORE);
    }
    return received;
}

MPIIslandCommunicator::~MPIIslandCommunicator() {
    for (auto& ps : pendingSends) {
        MPI_Cancel(&ps.request);
        MPI_Request_free(&ps.request);
    }
    MPI_Cancel(&recvRequest);
    MPI_Request_free(&recvRequest);
}

#endif // USE_MPI
