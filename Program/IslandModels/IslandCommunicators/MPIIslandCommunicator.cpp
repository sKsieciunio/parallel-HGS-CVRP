#include "MPIIslandCommunicator.h"
#include <limits>

#ifdef USE_MPI

void MPIIslandCommunicator::cleanupPendingSends() 
{
    pendingSends.remove_if([](PendingSend& ps) {
        int flag = 0;
        MPI_Test(&ps.request, &flag, MPI_STATUS_IGNORE);
        return flag;
        });
}

void MPIIslandCommunicator::postReceive() 
{
    MPI_Irecv(recvBuffer.data(), params.nbClients, MPI_INT, MPI_ANY_SOURCE, TAG_MIGRANT, MPI_COMM_WORLD, &recvRequest);
}

MPIIslandCommunicator::MPIIslandCommunicator(Params& params) : params(params) 
{
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    recvBuffer.resize(params.nbClients);
    postReceive();
}

void MPIIslandCommunicator::sendMigrants(const std::vector<Individual*>& migrants, const std::vector<int>& destinations) 
{
    cleanupPendingSends();
    for (const Individual* migrant : migrants) 
    {
        for (int dest : destinations) 
        {
            pendingSends.push_back({ migrant->chromT, MPI_REQUEST_NULL });
            MPI_Isend(pendingSends.back().buffer.data(), params.nbClients, MPI_INT, dest, TAG_MIGRANT, MPI_COMM_WORLD, &pendingSends.back().request);
        }
    }
}

std::vector<Individual> MPIIslandCommunicator::tryReceiveMigrants() 
{
    std::vector<Individual> received;
    int flag = 0;
    MPI_Test(&recvRequest, &flag, MPI_STATUS_IGNORE);
    while (flag) 
    {
        std::vector<int> chromT(recvBuffer.begin(), recvBuffer.end());
        received.emplace_back(params, chromT);
        postReceive();
        flag = 0;
        MPI_Test(&recvRequest, &flag, MPI_STATUS_IGNORE);
    }
    return received;
}

MPIIslandCommunicator::~MPIIslandCommunicator()
{
    for (auto& ps : pendingSends) 
    {
        MPI_Cancel(&ps.request);
        MPI_Request_free(&ps.request);
    }
    MPI_Cancel(&recvRequest);
    MPI_Request_free(&recvRequest);
}

bool MPIIslandCommunicator::getBestSolution(const Individual* bestLocal, int nbClients, std::vector<int>& outBestChromT, double& outBestCost) 
{
    double localCost = bestLocal ? bestLocal->eval.penalizedCost : std::numeric_limits<double>::max() / 2;

    Result localVal = { localCost, rank };
    Result globalVal;
    MPI_Allreduce(&localVal, &globalVal, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);

    outBestCost = globalVal.cost;

    if (globalVal.rank == rank && rank != 0) 
    {
        if (bestLocal) 
        {
            MPI_Send(bestLocal->chromT.data(), nbClients, MPI_INT, 0, TAG_BEST, MPI_COMM_WORLD);
        }
        else 
        {
            std::vector<int> empty(nbClients, 0);
            MPI_Send(empty.data(), nbClients, MPI_INT, 0, TAG_BEST, MPI_COMM_WORLD);
        }
    }

    if (rank == 0)
    {
        outBestChromT.resize(nbClients);
        if (globalVal.rank == 0) 
        {
            if (bestLocal)
                outBestChromT = bestLocal->chromT;
        }
        else 
        {
            MPI_Recv(outBestChromT.data(), nbClients, MPI_INT, globalVal.rank, TAG_BEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        return true;
    }
    return false;
}

#endif // USE_MPI
