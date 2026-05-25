#include "OpenMPLocalSearch.h"
#include "LocalSearch.h"

OpenMPLocalSearch::OpenMPLocalSearch(Params& params) : params(params)
{
	nThreads = params.ap.omplsnt;
	threadBestInsert.resize(nThreads,
		std::vector<std::vector<ThreeBestInsert>>(
			params.nbVehicles,
			std::vector<ThreeBestInsert>(params.nbClients + 1)));
	int maxPairs = params.nbVehicles * (params.nbVehicles - 1) / 2;
	qualifyingPairs.reserve(maxPairs);
	results.reserve(maxPairs);
}

bool OpenMPLocalSearch::runSwapStar(
	std::vector<Route>& routes,
	std::vector<Node>& clients,
	std::vector<Node>& depots,
	std::vector<int>& orderRoutes,
	int loopID, int nbMoves,
	double penCapacity, double penDuration)
{
	this->penaltyCapacityLS = penCapacity;
	this->penaltyDurationLS = penDuration;
	results.clear();

	// Build list of qualifying route pairs
	qualifyingPairs.clear();
	for (int rU = 0; rU < params.nbVehicles; rU++)
	{
		Route& ru = routes[orderRoutes[rU]];
		int lastTest = ru.whenLastTestedSWAPStar;
		ru.whenLastTestedSWAPStar = nbMoves;
		for (int rV = 0; rV < params.nbVehicles; rV++)
		{
			Route& rv = routes[orderRoutes[rV]];
			if (ru.nbCustomers > 0 && rv.nbCustomers > 0
				&& ru.cour < rv.cour
				&& (loopID == 0 || std::max<int>(ru.whenLastModified, rv.whenLastModified) > lastTest)
				&& CircleSector::overlap(ru.sector, rv.sector))
			{
				qualifyingPairs.push_back({orderRoutes[rU], orderRoutes[rV]});
			}
		}
	}

	if (qualifyingPairs.empty())
	{
		return false;
	}

	// Precompute deltaRemoval for all clients
	for (int r = 0; r < params.nbVehicles; r++)
	{
		for (Node* nd = routes[r].depot->next; !nd->isDepot; nd = nd->next)
		{
			nd->deltaRemoval = params.timeCost[nd->prev->cour][nd->next->cour]
				- params.timeCost[nd->prev->cour][nd->cour]
				- params.timeCost[nd->cour][nd->next->cour];
		}
	}

	// Evaluate all pairs in parallel
	results.resize(qualifyingPairs.size());

#pragma omp parallel for schedule(dynamic) num_threads(nThreads)
	for (int p = 0; p < (int)qualifyingPairs.size(); p++)
	{
		int tid = omp_get_thread_num();
		results[p] = evaluatePair(
			qualifyingPairs[p].rU, qualifyingPairs[p].rV,
			routes, clients, depots,
			threadBestInsert[tid]);
	}

	// Check if any pair found an improving move
	bool hasImproving = false;
	for (auto& r : results)
	{
		if (r.moveCost < -MY_EPSILON)
		{
			hasImproving = true;
			break;
		}
	}

	if (!hasImproving) return false;

	// Sort by cost
	std::sort(results.begin(), results.end(),
		[](const OmpSwapStarResult& a, const OmpSwapStarResult& b)
		{ return a.moveCost < b.moveCost; });

	int numResults = 0;
	for (auto& r : results)
	{
		if (r.moveCost >= -MY_EPSILON) break;
		numResults++;
	}
	results.resize(numResults);
	return true;

}

OmpSwapStarResult OpenMPLocalSearch::evaluatePair(
	int rU, int rV,
	std::vector<Route>& routes,
	std::vector<Node>& clients,
	std::vector<Node>& depots,
	std::vector<std::vector<ThreeBestInsert>>& localBestInsert)
{
	Route* myRouteU = &routes[rU];
	Route* myRouteV = &routes[rV];
	OmpSwapStarResult bestResult;
	bestResult.routeUIdx = rU;
	bestResult.routeVIdx = rV;

	// --- Preprocess insertions: U clients into routeV ---
	for (Node* U = myRouteU->depot->next; !U->isDepot; U = U->next)
	{
		ThreeBestInsert& bi = localBestInsert[rV][U->cour];
		bi.reset();
		bi.bestCost[0] = params.timeCost[0][U->cour]
			+ params.timeCost[U->cour][myRouteV->depot->next->cour]
			- params.timeCost[0][myRouteV->depot->next->cour];
		bi.bestLocation[0] = myRouteV->depot;
		for (Node* V = myRouteV->depot->next; !V->isDepot; V = V->next)
		{
			double dc = params.timeCost[V->cour][U->cour]
				+ params.timeCost[U->cour][V->next->cour]
				- params.timeCost[V->cour][V->next->cour];
			bi.compareAndAdd(dc, V);
		}
	}

	// --- Preprocess insertions: V clients into routeU ---
	for (Node* V = myRouteV->depot->next; !V->isDepot; V = V->next)
	{
		ThreeBestInsert& bi = localBestInsert[rU][V->cour];
		bi.reset();
		bi.bestCost[0] = params.timeCost[0][V->cour]
			+ params.timeCost[V->cour][myRouteU->depot->next->cour]
			- params.timeCost[0][myRouteU->depot->next->cour];
		bi.bestLocation[0] = myRouteU->depot;
		for (Node* U = myRouteU->depot->next; !U->isDepot; U = U->next)
		{
			double dc = params.timeCost[U->cour][V->cour]
				+ params.timeCost[V->cour][U->next->cour]
				- params.timeCost[U->cour][U->next->cour];
			bi.compareAndAdd(dc, U);
		}
	}

	// --- Lambda: getCheapestInsertSimultRemoval using local buffer ---
	auto getCheapest = [&](Node* ins, Node* rem, int remRouteIdx, Node*& bestPos) -> double
		{
			ThreeBestInsert& bi = localBestInsert[remRouteIdx][ins->cour];
			bestPos = bi.bestLocation[0];
			double best = bi.bestCost[0];
			bool found = (bestPos != rem && bestPos->next != rem);

			if (!found && bi.bestLocation[1] != nullptr)
			{
				bestPos = bi.bestLocation[1];
				best = bi.bestCost[1];
				found = (bestPos != rem && bestPos->next != rem);
				if (!found && bi.bestLocation[2] != nullptr)
				{
					bestPos = bi.bestLocation[2];
					best = bi.bestCost[2];
					found = true;
				}
			}

			double dc = params.timeCost[rem->prev->cour][ins->cour]
				+ params.timeCost[ins->cour][rem->next->cour]
				- params.timeCost[rem->prev->cour][rem->next->cour];
			if (!found || dc < best)
			{
				bestPos = rem->prev;
				best = dc;
			}
			return best;
		};

	// --- Evaluate SWAP* moves ---
	for (Node* U = myRouteU->depot->next; !U->isDepot; U = U->next)
	{
		for (Node* V = myRouteV->depot->next; !V->isDepot; V = V->next)
		{
			double deltaPenU = penaltyExcessLoad(myRouteU->load + params.cli[V->cour].demand - params.cli[U->cour].demand) - myRouteU->penalty;
			double deltaPenV = penaltyExcessLoad(myRouteV->load + params.cli[U->cour].demand - params.cli[V->cour].demand) - myRouteV->penalty;

			if (deltaPenU + U->deltaRemoval + deltaPenV + V->deltaRemoval <= 0)
			{
				OmpSwapStarResult candidate;
				candidate.U = U;
				candidate.V = V;
				candidate.routeUIdx = rU;
				candidate.routeVIdx = rV;

				double extraV = getCheapest(U, V, rV, candidate.bestPositionU);
				double extraU = getCheapest(V, U, rU, candidate.bestPositionV);

				candidate.moveCost = deltaPenU + U->deltaRemoval + extraU + deltaPenV + V->deltaRemoval + extraV
					+ penaltyExcessDuration(myRouteU->duration + U->deltaRemoval + extraU + params.cli[V->cour].serviceDuration - params.cli[U->cour].serviceDuration)
					+ penaltyExcessDuration(myRouteV->duration + V->deltaRemoval + extraV - params.cli[V->cour].serviceDuration + params.cli[U->cour].serviceDuration);

				if (candidate.moveCost < bestResult.moveCost)
					bestResult = candidate;
			}
		}
	}

	// --- Evaluate RELOCATE U -> routeV ---
	for (Node* U = myRouteU->depot->next; !U->isDepot; U = U->next)
	{
		OmpSwapStarResult candidate;
		candidate.U = U;
		candidate.routeUIdx = rU;
		candidate.routeVIdx = rV;
		candidate.bestPositionU = localBestInsert[rV][U->cour].bestLocation[0];

		double deltaDistU = params.timeCost[U->prev->cour][U->next->cour]
			- params.timeCost[U->prev->cour][U->cour]
			- params.timeCost[U->cour][U->next->cour];
		double deltaDistV = localBestInsert[rV][U->cour].bestCost[0];

		candidate.moveCost = deltaDistU + deltaDistV
			+ penaltyExcessLoad(myRouteU->load - params.cli[U->cour].demand) - myRouteU->penalty
			+ penaltyExcessLoad(myRouteV->load + params.cli[U->cour].demand) - myRouteV->penalty
			+ penaltyExcessDuration(myRouteU->duration + deltaDistU - params.cli[U->cour].serviceDuration)
			+ penaltyExcessDuration(myRouteV->duration + deltaDistV + params.cli[U->cour].serviceDuration);

		if (candidate.moveCost < bestResult.moveCost)
			bestResult = candidate;
	}

	// --- Evaluate RELOCATE V -> routeU ---
	for (Node* V = myRouteV->depot->next; !V->isDepot; V = V->next)
	{
		OmpSwapStarResult candidate;
		candidate.V = V;
		candidate.routeUIdx = rU;
		candidate.routeVIdx = rV;
		candidate.bestPositionV = localBestInsert[rU][V->cour].bestLocation[0];

		double deltaDistU = localBestInsert[rU][V->cour].bestCost[0];
		double deltaDistV = params.timeCost[V->prev->cour][V->next->cour]
			- params.timeCost[V->prev->cour][V->cour]
			- params.timeCost[V->cour][V->next->cour];

		candidate.moveCost = deltaDistU + deltaDistV
			+ penaltyExcessLoad(myRouteU->load + params.cli[V->cour].demand) - myRouteU->penalty
			+ penaltyExcessLoad(myRouteV->load - params.cli[V->cour].demand) - myRouteV->penalty
			+ penaltyExcessDuration(myRouteU->duration + deltaDistU + params.cli[V->cour].serviceDuration)
			+ penaltyExcessDuration(myRouteV->duration + deltaDistV - params.cli[V->cour].serviceDuration);

		if (candidate.moveCost < bestResult.moveCost)
			bestResult = candidate;
	}

	return bestResult;
}