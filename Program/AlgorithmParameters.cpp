//
// Created by chkwon on 3/23/22.
//

#include "AlgorithmParameters.h"
#include <iostream>
#include <omp.h>

extern "C"
struct AlgorithmParameters default_algorithm_parameters() {
	struct AlgorithmParameters ap{};

	ap.nbGranular = 20;
	ap.mu = 25;
	ap.lambda = 40;
	ap.nbElite = 4;
	ap.nbClose = 5;

	ap.nbIterPenaltyManagement = 100;
	ap.targetFeasible = 0.2;
	ap.penaltyDecrease = 0.85;
	ap.penaltyIncrease = 1.2;

	ap.seed = 0;
	ap.nbIter = 20000;
	ap.nbIterTraces = 500;
	ap.timeLimit = 0;
	ap.useSwapStar = 1;
	ap.useGpu = 0;

	ap.useOpenMp = 0;
	ap.omplsnt = omp_get_max_threads();

	ap.useIslandModel = 0;

	ap.topology = 0;
	ap.nbNodes = 1;

	ap.migrationPolicy = 0;
	ap.interval = 20;
	ap.warmup = 20;
	ap.sendCooldown = 20;
	ap.receiveStagnationThreshold = 20;
	ap.minReceiveInterval = 20;
	ap.maxReceiveInterval = 100;

	ap.immigrantHandler = 0;

	ap.migrantSelector = 0;
	ap.selectionCount = 1;

	ap.islandCommunicator = 0;

	ap.makeManyOffspring = 0;
	ap.numOffspring = omp_get_max_threads();
	ap.numThreadsOffspring = omp_get_max_threads();

	return ap;
}

void print_algorithm_parameters(const AlgorithmParameters & ap)
{
	std::cout << "=========== Algorithm Parameters =================" << std::endl;
	std::cout << "---- nbGranular              is set to " << ap.nbGranular << std::endl;
	std::cout << "---- mu                      is set to " << ap.mu << std::endl;
	std::cout << "---- lambda                  is set to " << ap.lambda << std::endl;
	std::cout << "---- nbElite                 is set to " << ap.nbElite << std::endl;
	std::cout << "---- nbClose                 is set to " << ap.nbClose << std::endl;
	std::cout << "---- nbIterPenaltyManagement is set to " << ap.nbIterPenaltyManagement << std::endl;
	std::cout << "---- targetFeasible          is set to " << ap.targetFeasible << std::endl;
	std::cout << "---- penaltyDecrease         is set to " << ap.penaltyDecrease << std::endl;
	std::cout << "---- penaltyIncrease         is set to " << ap.penaltyIncrease << std::endl;
	std::cout << "---- seed                    is set to " << ap.seed << std::endl;
	std::cout << "---- nbIter                  is set to " << ap.nbIter << std::endl;
	std::cout << "---- nbIterTraces            is set to " << ap.nbIterTraces << std::endl;
	std::cout << "---- timeLimit               is set to " << ap.timeLimit << std::endl;
	std::cout << "---- useSwapStar             is set to " << ap.useSwapStar << std::endl;
	std::cout << "---- useGpu                  is set to " << ap.useGpu << std::endl;
	std::cout << "---- useOpenMp               is set to " << ap.useOpenMp << std::endl;
	std::cout << "---- omplsnt				   is set to " << ap.omplsnt << std::endl;					
	std::cout << "---- makeManyOffspring	   is set to " << ap.makeManyOffspring << std::endl;
	std::cout << "---- numOffspring			   is set to " << ap.numOffspring << std::endl;
	std::cout << "---- numThreadsOffspring	   is set to " << ap.numThreadsOffspring << std::endl;
	std::cout << "==================================================" << std::endl;

	std::cout << "=========== Island Model Parameters ==============" << std::endl;
	std::cout << "---- useIslandModel					   is set to " << ap.useIslandModel << std::endl;
	std::cout << "---- topology							   is set to " << ap.topology << std::endl;
	std::cout << "---- nbNodes							   is set to " << ap.nbNodes << std::endl;
	std::cout << "---- migrationPolicy					   is set to " << ap.migrationPolicy << std::endl;
	std::cout << "---- interval							   is set to " << ap.interval << std::endl;
	std::cout << "---- warmup							   is set to " << ap.warmup << std::endl;
	std::cout << "---- sendCooldown						   is set to " << ap.sendCooldown << std::endl;
	std::cout << "---- receiveStagnationThreshold		   is set to " << ap.receiveStagnationThreshold << std::endl;
	std::cout << "---- minReceiveInterval				   is set to " << ap.minReceiveInterval << std::endl;
	std::cout << "---- maxReceiveInterval				   is set to " << ap.maxReceiveInterval << std::endl;
	std::cout << "---- migrantSelector					   is set to " << ap.migrantSelector << std::endl;
	std::cout << "---- selectionCount					   is set to " << ap.selectionCount << std::endl;
	std::cout << "---- islandCommunicator				   is set to " << ap.islandCommunicator << std::endl;
	std::cout << "==================================================" << std::endl;
}
