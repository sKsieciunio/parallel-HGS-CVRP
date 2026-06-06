#ifndef OPENMP_LOCAL_SEARCH_H
#define OPENMP_LOCAL_SEARCH_H

#include "Params.h"
#include "CircleSector.h"
#include <vector>
#include <algorithm>
#include <omp.h>

struct Node;
struct Route;
struct ThreeBestInsert;
struct SwapStarElement;

struct RoutePair 
{
	int rU;
	int rV;
};

// Result of a single SWAP* pair evaluation
struct OmpSwapStarResult
{
	double moveCost = 1.e30;
	Node* U = nullptr;
	Node* bestPositionU = nullptr;
	Node* V = nullptr;
	Node* bestPositionV = nullptr;
	int routeUIdx = -1;
	int routeVIdx = -1;
};

class OpenMPLocalSearch
{
private:
	Params& params;
	int nThreads;
	std::vector<RoutePair> qualifyingPairs;

	// Per-thread buffers for ThreeBestInsert: [nThreads][2][nbClients+1]
	// Index 0 = buffer for routeU, index 1 = buffer for routeV.
	// Two rows per thread instead of nbVehicles rows — avoids GBs of allocation
	// for large instances (XL-n10001 with nbVehicles~1570 and 8 threads = ~6 GB).
	std::vector<std::vector<std::vector<ThreeBestInsert>>> threadBestInsert;

	// Evaluate a single (routeU, routeV) pair
	OmpSwapStarResult evaluatePair(
		int rU, int rV,
		std::vector<Route>& routes,
		std::vector<Node>& clients,
		std::vector<Node>& depots,
		std::vector<std::vector<ThreeBestInsert>>& localBestInsert);

	// Helpers matching LocalSearch penalty functions
	double penaltyCapacityLS;
	double penaltyDurationLS;
	inline double penaltyExcessDuration(double d) const {
		return std::max<double>(0., d - params.durationLimit) * penaltyDurationLS;
	}
	inline double penaltyExcessLoad(double l) const {
		return std::max<double>(0., l - params.vehicleCapacity) * penaltyCapacityLS;
	}

public:
	OpenMPLocalSearch(Params& params);

	// Collected results sorted by cost
	std::vector<OmpSwapStarResult> results;

	// Evaluate all qualifying route pairs in parallel, populate `results`.
	// Returns true if at least one improving move was found.
	bool runSwapStar(
		std::vector<Route>& routes,
		std::vector<Node>& clients,
		std::vector<Node>& depots,
		std::vector<int>& orderRoutes,
		int loopID, int nbMoves,
		double penaltyCapacityLS,
		double penaltyDurationLS);
};

#endif // OPENMP_LOCAL_SEARCH_H