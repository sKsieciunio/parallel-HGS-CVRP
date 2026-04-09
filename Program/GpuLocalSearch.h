// GpuLocalSearch.h
// Pure C++ interface to the GPU local search implementation.
// No CUDA types — safe to include from any .cpp translation unit.
#pragma once
#include "GpuDataLayout.cuh"

// Opaque handle — full class definition lives only in GpuLocalSearch.cu.
class GpuLocalSearch;

// Allocates GPU memory and uploads the problem's static data (distance matrix,
// client demands and service durations).  Returns nullptr on any CUDA failure.
//   timeCostFlat : row-major (nbClients+1) x (nbClients+1) distance matrix.
//   demand       : client demands, length nbClients+1 (index 0 = depot).
//   service      : client service durations, same length.
GpuLocalSearch* createGpuLocalSearch(
    const double* timeCostFlat,
    const double* demand,
    const double* service,
    int    nbClients,
    int    nbVehicles,
    double vehicleCapacity,
    double durationLimit
);

// Frees all device memory owned by the handle.
void destroyGpuLocalSearch(GpuLocalSearch* handle);

// Evaluates SWAP* for every supplied route pair on the GPU in one kernel launch.
// The caller is responsible for flattening the linked-list representation into
// contiguous arrays and selecting the qualifying (sector-overlapping) pairs.
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
//   result (out)       : globally best improving move across all pairs.
//
// Evaluates SWAP* for all supplied route pairs on the GPU in one kernel launch.
// Writes one GpuSwapStarResult per pair into outResults[0..numPairs-1] (caller-allocated).
// Returns true if at least one pair has an improving move (moveCost < -epsilon).
bool gpuEvaluateSwapStar(
    GpuLocalSearch* handle,
    const int*    routeStart,
    const int*    routeLen,
    const double* routeDuration,
    const double* routeLoad,
    const double* routePenalty,
    const int*    routeCustomers,
    const double* deltaRemoval,
    const int*    routePairU,
    const int*    routePairV,
    int           numPairs,
    double        penaltyCapacity,
    double        penaltyDuration,
    GpuSwapStarResult* outResults   // [numPairs] — written by this function
);
