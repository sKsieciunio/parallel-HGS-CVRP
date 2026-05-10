#include "Genetic.h"
#include "commandline.h"
#include "LocalSearch.h"
#include "Split.h"
#include "InstanceCVRPLIB.h"
#include "CudaSanity.h"
#ifdef USE_MPI
#include <mpi.h>
#endif // USE_MPI

using namespace std;

int main(int argc, char *argv[])
{
#ifdef USE_MPI
	MPI_Init(&argc, &argv);
#endif

	try
	{
		if (!runCudaSanityCheck())
		{
			std::cout << "CUDA sanity check failed. Continuing on CPU path." << std::endl;
		}

		// Reading the arguments of the program
		CommandLine commandline(argc, argv);

		// Print all algorithm parameter values
		if (commandline.verbose)
			print_algorithm_parameters(commandline.ap);

		// Reading the data file and initializing some data structures
		if (commandline.verbose)
			std::cout << "----- READING INSTANCE: " << commandline.pathInstance << std::endl;
		InstanceCVRPLIB cvrp(commandline.pathInstance, commandline.isRoundingInteger);

		Params params(cvrp.x_coords, cvrp.y_coords, cvrp.dist_mtx, cvrp.service_time, cvrp.demands,
					  cvrp.vehicleCapacity, cvrp.durationLimit, commandline.nbVeh, cvrp.isDurationConstraint, commandline.verbose, commandline.ap);

		
        if (commandline.ap.useIslandModel)
        {
            IslandModel islandModel(params);

			int islandRank = islandModel.islandCommunicator->getRank();
			params.ran.seed(commandline.ap.seed + islandRank);

            Genetic solver(params, islandModel);
            solver.run();

            Individual bestGlobal(params);
            if (islandModel.getBestSolution(solver.population, solver.split, params, bestGlobal))
            {
                if (params.verbose)
                    std::cout << "----- BEST GLOBAL COST: " << bestGlobal.eval.penalizedCost << std::endl;
                solver.population.exportCVRPLibFormat(bestGlobal, commandline.pathSolution);
                solver.population.exportSearchProgress(commandline.pathSolution + ".PG.csv", commandline.pathInstance);
            }
        }
        else
        {
            Genetic solver(params);
            solver.run();
            if (solver.population.getBestFound() != NULL) {
                solver.population.exportCVRPLibFormat(*solver.population.getBestFound(), commandline.pathSolution);
                solver.population.exportSearchProgress(commandline.pathSolution + ".PG.csv", commandline.pathInstance);
            }
        }
	}
	catch (const string &e)
	{
		std::cout << "EXCEPTION | " << e << std::endl;
	}
	catch (const std::exception &e)
	{
		std::cout << "EXCEPTION | " << e.what() << std::endl;
	}

#ifdef USE_MPI
	MPI_Finalize();
#endif

	return 0;
}
