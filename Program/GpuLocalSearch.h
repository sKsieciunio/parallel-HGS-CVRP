// GpuLocalSearch.h
// Pure C++ interface to the GPU local search implementation.
// No CUDA types — safe to include from any .cpp translation unit.
#pragma once
#include "GpuDataLayout.cuh"

// Opaque handles — full definitions live only in GpuLocalSearch.cu.
class GpuLocalSearch;
struct GpuProblemData;

// ---------------------------------------------------------------------------
// Shared, read-only problem data (distance matrix, demands, service times).
// Upload ONCE per process (or per island/rank) and reference it from every
// per-thread GpuLocalSearch handle, instead of duplicating the (nbClients+1)^2
// distance matrix in each handle.
//   timeCostFlat : row-major (nbClients+1) x (nbClients+1) distance matrix.
//   demand       : client demands, length nbClients+1 (index 0 = depot).
//   service      : client service durations, same length.
// Returns nullptr on any CUDA failure.
GpuProblemData * createGpuProblemData(
	const double * timeCostFlat,
	const double * demand,
	const double * service,
	int nbClients);

void destroyGpuProblemData(GpuProblemData * handle);

// Allocates per-thread working buffers + a private CUDA stream, referencing the
// shared problem data (NOT owned by the returned handle). Returns nullptr on
// any CUDA failure.
GpuLocalSearch * createGpuLocalSearchShared(
	GpuProblemData * shared,
	int nbClients,
	int nbVehicles,
	double vehicleCapacity,
	double durationLimit);

// Legacy all-in-one factory: uploads its own private copy of the static problem
// data and wraps a single handle around it. Kept for backward compatibility;
// prefer createGpuProblemData + createGpuLocalSearchShared to share static data.
GpuLocalSearch * createGpuLocalSearch(
	const double * timeCostFlat,
	const double * demand,
	const double * service,
	int nbClients,
	int nbVehicles,
	double vehicleCapacity,
	double durationLimit);

// Frees per-instance device memory + stream owned by the handle (and the shared
// problem data only if this handle created it via the legacy factory).
void destroyGpuLocalSearch(GpuLocalSearch * handle);

// Evaluates SWAP* for all supplied route pairs on the GPU in one kernel launch.
// Writes one GpuSwapStarResult per pair into outResults[0..numPairs-1] (caller-allocated).
// Returns true if at least one pair has an improving move (moveCost < -epsilon);
// returns false on no improvement OR on any CUDA error (caller keeps CPU state).
//
//   routeStart[r]      : offset into routeCustomers where route r begins.
//   routeLen[r]        : number of customers in route r.
//   routeDuration[r]   : current total duration of route r.
//   routeLoad[r]       : current total load of route r.
//   routePenalty[r]    : current penalty (capacity + duration) of route r.
//   routeCustomers[i]  : customer IDs, packed contiguously (all routes).
//   deltaRemoval[i]    : marginal removal cost for routeCustomers[i].
//   routePairU/V[p]    : route indices for pair p (rU < rV, both non-empty).
//   numPairs           : number of pairs to evaluate.
//   penaltyCapacity/Duration : current penalty coefficients.
bool gpuEvaluateSwapStar(
	GpuLocalSearch * handle,
	const int * routeStart,
	const int * routeLen,
	const double * routeDuration,
	const double * routeLoad,
	const double * routePenalty,
	const int * routeCustomers,
	const double * deltaRemoval,
	const int * routePairU,
	const int * routePairV,
	int numPairs,
	double penaltyCapacity,
	double penaltyDuration,
	GpuSwapStarResult * outResults // [numPairs] — written by this function
);