#include "SynchronousMPIIslandCommunicator.h"
#include <limits>

#ifdef USE_MPI

SynchronousMPIIslandCommunicator::SynchronousMPIIslandCommunicator(Params& params) : params(params) 
{
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    recvBuf.resize(params.nbClients);
}

void SynchronousMPIIslandCommunicator::sendMigrants(const std::vector<Individual*>& migrants,const std::vector<int>& destinations) 
{
    sendRequests.clear();
    sendBuffers.clear();

    if (!migrants.empty() && !destinations.empty()) 
    {
        for (int dest : destinations) 
        {
            sendBuffers.push_back(migrants[0]->chromT);
            sendRequests.push_back(MPI_REQUEST_NULL);
            MPI_Isend(sendBuffers.back().data(), params.nbClients, MPI_INT, dest, TAG_MIGRANT, MPI_COMM_WORLD, &sendRequests.back());
        }
    }

    if (!sendRequests.empty())
    {
        MPI_Waitall(sendRequests.size(), sendRequests.data(), MPI_STATUSES_IGNORE);
    }
}

std::vector<Individual> SynchronousMPIIslandCommunicator::tryReceiveMigrants() 
{
    receivedBuffer.clear();

    int flag = 0;
    MPI_Iprobe(MPI_ANY_SOURCE, TAG_MIGRANT, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);

    while (flag) 
    {
        MPI_Recv(recvBuf.data(), params.nbClients, MPI_INT, MPI_ANY_SOURCE, TAG_MIGRANT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        receivedBuffer.emplace_back(params, recvBuf);

        flag = 0;
        MPI_Iprobe(MPI_ANY_SOURCE, TAG_MIGRANT, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
    }

    return std::move(receivedBuffer);
}

bool SynchronousMPIIslandCommunicator::getBestSolution(const Individual* bestLocal, int nbClients, std::vector<int>& outBestChromT, double& outBestCost) 
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

SynchronousMPIIslandCommunicator::~SynchronousMPIIslandCommunicator()
{
	int flag = 0;
	MPI_Iprobe(MPI_ANY_SOURCE, TAG_MIGRANT, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
	while (flag)
    {
		MPI_Recv(recvBuf.data(), params.nbClients, MPI_INT, MPI_ANY_SOURCE, TAG_MIGRANT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		flag = 0;
		MPI_Iprobe(MPI_ANY_SOURCE, TAG_MIGRANT, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
	}
}

#endif // USE_MPI