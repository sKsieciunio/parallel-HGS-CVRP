#include "Genetic.h"
#include "commandline.h"
#include "LocalSearch.h"
#include "Split.h"
#include "InstanceCVRPLIB.h"
#include "CudaSanity.h"
#include "HardwareMonitor.h"
#include <sstream>
#include <iomanip>
#ifdef USE_MPI
#include <mpi.h>
#endif // USE_MPI

using namespace std;

static void printHardwareSummary(const HardwareMonitor &hw)
{
    std::cout << "\n  -- Hardware utilization (" << hw.sampleCount() << " samples @ 500ms) --\n";
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "  CPU avg util : " << hw.avgCpuUtil() << " %\n";
    if (hw.gpuMonEnabled())
    {
        std::cout << "  GPU core avg : " << hw.avgGpuCoreUtil() << " %\n";
        std::cout << "  GPU mem  avg : " << hw.avgGpuMemUtil() << " %\n";
    }
    else
    {
        std::cout << "  GPU monitoring: disabled (NVML not available)\n";
    }
    std::cout << "=======================================================\n\n";
}

static std::string buildModeString(const AlgorithmParameters &ap)
{
    std::string mode;
    if (ap.useIslandModel)
    {
        mode += "island" + std::to_string(ap.nbNodes);
        const char *topoNames[] = {"ring", "2ring", "hypercube", "star", "full", "randReg"};
        if (ap.topology >= 0 && ap.topology < 6)
            mode += std::string("+") + topoNames[ap.topology];
        const char *polNames[] = {"fixed", "improve", "adaptive", "diversity"};
        if (ap.migrationPolicy >= 0 && ap.migrationPolicy < 4)
            mode += std::string("+") + polNames[ap.migrationPolicy];
    }
    else
    {
        mode = "single";
    }
    if (ap.useGpu)
        mode += "+gpu";
    if (ap.useOpenMp)
        mode += "+omp";
    if (ap.makeManyOffspring)
        mode += "+offspring" + std::to_string(ap.numOffspring) + "t" + std::to_string(ap.numThreadsOffspring);
    return mode;
}

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

        std::string mode = buildModeString(commandline.ap);

        // Hardware monitor — one per process (each MPI rank monitors its own GPU/CPU)
        HardwareMonitor hwMonitor(0, 500);
        hwMonitor.start();

        if (commandline.ap.useIslandModel)
        {
            IslandModel islandModel(params);

            int islandRank = islandModel.islandCommunicator->getRank();
            params.ap.seed += islandRank;
            params.ran.seed(params.ap.seed);
            params.convergenceCsvPath = commandline.pathSolution + ".conv.r" + std::to_string(islandRank) + ".csv";

            Genetic solver(params, islandModel);
            solver.run();
            hwMonitor.stop();

            Individual bestGlobal(params);
            if (islandModel.getBestSolution(solver.population, solver.split, params, bestGlobal))
            {
                if (params.verbose)
                    std::cout << "----- BEST GLOBAL COST: " << bestGlobal.eval.penalizedCost << std::endl;
                solver.population.exportCVRPLibFormat(bestGlobal, commandline.pathSolution);
                solver.population.exportSearchProgress(commandline.pathSolution + ".PG.csv", commandline.pathInstance);

                double elapsed = (double)(clock() - params.startTime) / (double)CLOCKS_PER_SEC;
                if (commandline.verbose)
                {
                    params.telemetry.printSummary(commandline.pathInstance, commandline.ap.seed, mode,
                                                  bestGlobal.eval.penalizedCost, elapsed);
                    printHardwareSummary(hwMonitor);
                }
                params.telemetry.exportCSV(commandline.pathSolution + ".tel.csv",
                                           commandline.pathInstance, commandline.ap.seed, mode,
                                           bestGlobal.eval.penalizedCost, elapsed, hwMonitor);
            }
        }
        else
        {
            params.convergenceCsvPath = commandline.pathSolution + ".conv.csv";
            Genetic solver(params);
            solver.run();
            hwMonitor.stop();

            if (solver.population.getBestFound() != NULL)
            {
                const Individual *best = solver.population.getBestFound();
                solver.population.exportCVRPLibFormat(*best, commandline.pathSolution);
                solver.population.exportSearchProgress(commandline.pathSolution + ".PG.csv", commandline.pathInstance);

                double elapsed = (double)(clock() - params.startTime) / (double)CLOCKS_PER_SEC;
                if (commandline.verbose)
                {
                    params.telemetry.printSummary(commandline.pathInstance, commandline.ap.seed, mode,
                                                  best->eval.penalizedCost, elapsed);
                    printHardwareSummary(hwMonitor);
                }
                params.telemetry.exportCSV(commandline.pathSolution + ".tel.csv",
                                           commandline.pathInstance, commandline.ap.seed, mode,
                                           best->eval.penalizedCost, elapsed, hwMonitor);
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
