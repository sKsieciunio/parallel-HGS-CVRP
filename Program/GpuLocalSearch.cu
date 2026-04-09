// GpuLocalSearch.cu
// Host-side orchestration: device memory management, data transfer, kernel
// dispatch, and result selection for GPU-accelerated SWAP* evaluation.
#include "GpuLocalSearch.h"
#include "GpuSwapStarKernel.cuh"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <vector>

// ---------------------------------------------------------------------------
// Error checking helpers
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s:%d — %s\n",                    \
                         __FILE__, __LINE__, cudaGetErrorString(_err));         \
        }                                                                       \
    } while (0)

#define CUDA_CHECK_RETURN(call)                                                 \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s:%d — %s\n",                    \
                         __FILE__, __LINE__, cudaGetErrorString(_err));         \
            return nullptr;                                                     \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Block size for SWAP* kernel (tunable)
// ---------------------------------------------------------------------------
static constexpr int BLOCK_SIZE = 256;

// ---------------------------------------------------------------------------
// GpuLocalSearch class — full definition (CUDA-aware)
// ---------------------------------------------------------------------------
class GpuLocalSearch
{
public:
    int    nbClients_;
    int    nbVehicles_;
    double vehicleCapacity_;
    double durationLimit_;

    // ---- Device buffers — problem data (uploaded once in constructor) ----
    double* d_timeCost_;       // (nbClients+1)^2  row-major
    double* d_demand_;         // nbClients+1
    double* d_service_;        // nbClients+1

    // ---- Device buffers — route data (re-uploaded each call) ----
    int*    d_routeStart_;     // nbVehicles
    int*    d_routeLen_;       // nbVehicles
    double* d_routeDuration_;  // nbVehicles
    double* d_routeLoad_;      // nbVehicles
    double* d_routePenalty_;   // nbVehicles
    int*    d_routeCustomers_; // nbClients  (packed, depot excluded)
    double* d_deltaRemoval_;   // nbClients

    // ---- Device buffers — pairs and results ----
    int*               d_pairU_;    // maxPairs
    int*               d_pairV_;    // maxPairs
    GpuSwapStarResult* d_results_;  // maxPairs

    int maxPairs_;

    GpuLocalSearch() = default;

    // Evaluates all route pairs in one kernel launch.
    // Writes one result per pair into outResults[0..numPairs-1].
    // Returns true if any pair has an improving move.
    bool evaluateSwapStar(
        const int*    routeStart,
        const int*    routeLen,
        const double* routeDuration,
        const double* routeLoad,
        const double* routePenalty,
        const int*    routeCustomers,
        const double* deltaRemoval,
        const int*    pairU,
        const int*    pairV,
        int           numPairs,
        double        penaltyCapacity,
        double        penaltyDuration,
        GpuSwapStarResult* outResults)
    {
        if (numPairs == 0) return false;

        // Upload route data
        CUDA_CHECK(cudaMemcpy(d_routeStart_,    routeStart,    nbVehicles_ * sizeof(int),    cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_routeLen_,      routeLen,      nbVehicles_ * sizeof(int),    cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_routeDuration_, routeDuration, nbVehicles_ * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_routeLoad_,     routeLoad,     nbVehicles_ * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_routePenalty_,  routePenalty,  nbVehicles_ * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_routeCustomers_,routeCustomers,nbClients_  * sizeof(int),    cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_deltaRemoval_,  deltaRemoval,  nbClients_  * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pairU_, pairU, numPairs * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pairV_, pairV, numPairs * sizeof(int), cudaMemcpyHostToDevice));

        // Compute max route length to size shared memory
        const int maxLen = *std::max_element(routeLen, routeLen + nbVehicles_);

        // Shared memory layout (see GpuSwapStarKernel.cuh for details):
        //   int[maxLen*2] + double[maxLen*2] + GpuSwapStarResult[BLOCK_SIZE]
        // Each section is 8-byte aligned.
        const size_t intsBytes  = ((size_t)(2 * maxLen) * sizeof(int)    + 7u) & ~7u;
        const size_t dblsBytes  =  (size_t)(2 * maxLen) * sizeof(double);
        const size_t reducBytes = ((intsBytes + dblsBytes + 7u) & ~7u)
                                  - (intsBytes + dblsBytes)
                                  + (size_t)BLOCK_SIZE * sizeof(GpuSwapStarResult);
        const size_t smemSize = intsBytes + dblsBytes + reducBytes;

        // One block per route pair — all pairs evaluated in parallel
        evalSwapStarKernel<<<numPairs, BLOCK_SIZE, smemSize>>>(
            d_routeStart_, d_routeLen_, d_routeDuration_, d_routeLoad_, d_routePenalty_,
            d_routeCustomers_, d_deltaRemoval_,
            d_timeCost_, d_demand_, d_service_,
            nbClients_, vehicleCapacity_, durationLimit_,
            penaltyCapacity, penaltyDuration,
            d_pairU_, d_pairV_, numPairs,
            d_results_
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Download all per-pair results in one transfer
        CUDA_CHECK(cudaMemcpy(outResults, d_results_,
                              numPairs * sizeof(GpuSwapStarResult),
                              cudaMemcpyDeviceToHost));

        // Report whether any pair found an improving move
        for (int i = 0; i < numPairs; i++)
            if (outResults[i].moveCost < -1e-5)
                return true;
        return false;
    }
};

// ---------------------------------------------------------------------------
// Public C++ interface (callable from non-CUDA translation units)
// ---------------------------------------------------------------------------

GpuLocalSearch* createGpuLocalSearch(
    const double* timeCostFlat,
    const double* demand,
    const double* service,
    int    nbClients,
    int    nbVehicles,
    double vehicleCapacity,
    double durationLimit)
{
    // Verify a CUDA device is available
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
    {
        std::fprintf(stderr, "GpuLocalSearch: no CUDA device available.\n");
        return nullptr;
    }

    GpuLocalSearch* ls = new GpuLocalSearch();
    ls->nbClients_      = nbClients;
    ls->nbVehicles_     = nbVehicles;
    ls->vehicleCapacity_ = vehicleCapacity;
    ls->durationLimit_   = durationLimit;

    const int n = nbClients + 1;
    const int maxPairs = nbVehicles * (nbVehicles - 1) / 2;
    ls->maxPairs_ = maxPairs;

    // ---- Allocate device memory ----
    CUDA_CHECK_RETURN(cudaMalloc(&ls->d_timeCost_,       (size_t)n * n           * sizeof(double)));
    CUDA_CHECK_RETURN(cudaMalloc(&ls->d_demand_,         (size_t)n               * sizeof(double)));
    CUDA_CHECK_RETURN(cudaMalloc(&ls->d_service_,        (size_t)n               * sizeof(double)));

    CUDA_CHECK_RETURN(cudaMalloc(&ls->d_routeStart_,     (size_t)nbVehicles      * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc(&ls->d_routeLen_,       (size_t)nbVehicles      * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc(&ls->d_routeDuration_,  (size_t)nbVehicles      * sizeof(double)));
    CUDA_CHECK_RETURN(cudaMalloc(&ls->d_routeLoad_,      (size_t)nbVehicles      * sizeof(double)));
    CUDA_CHECK_RETURN(cudaMalloc(&ls->d_routePenalty_,   (size_t)nbVehicles      * sizeof(double)));
    CUDA_CHECK_RETURN(cudaMalloc(&ls->d_routeCustomers_, (size_t)nbClients       * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc(&ls->d_deltaRemoval_,   (size_t)nbClients       * sizeof(double)));

    CUDA_CHECK_RETURN(cudaMalloc(&ls->d_pairU_,   (size_t)maxPairs * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc(&ls->d_pairV_,   (size_t)maxPairs * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc(&ls->d_results_, (size_t)maxPairs * sizeof(GpuSwapStarResult)));

    // ---- Upload static problem data ----
    CUDA_CHECK_RETURN(cudaMemcpy(ls->d_timeCost_, timeCostFlat,
                                 (size_t)n * n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(ls->d_demand_,  demand,
                                 (size_t)n       * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(ls->d_service_, service,
                                 (size_t)n       * sizeof(double), cudaMemcpyHostToDevice));

    return ls;
}

void destroyGpuLocalSearch(GpuLocalSearch* ls)
{
    if (!ls) return;
    cudaFree(ls->d_timeCost_);
    cudaFree(ls->d_demand_);
    cudaFree(ls->d_service_);
    cudaFree(ls->d_routeStart_);
    cudaFree(ls->d_routeLen_);
    cudaFree(ls->d_routeDuration_);
    cudaFree(ls->d_routeLoad_);
    cudaFree(ls->d_routePenalty_);
    cudaFree(ls->d_routeCustomers_);
    cudaFree(ls->d_deltaRemoval_);
    cudaFree(ls->d_pairU_);
    cudaFree(ls->d_pairV_);
    cudaFree(ls->d_results_);
    delete ls;
}

bool gpuEvaluateSwapStar(
    GpuLocalSearch* handle,
    const int*    routeStart,
    const int*    routeLen,
    const double* routeDuration,
    const double* routeLoad,
    const double* routePenalty,
    const int*    routeCustomers,
    const double* deltaRemoval,
    const int*    pairU,
    const int*    pairV,
    int           numPairs,
    double        penaltyCapacity,
    double        penaltyDuration,
    GpuSwapStarResult* outResults)
{
    return handle->evaluateSwapStar(
        routeStart, routeLen, routeDuration, routeLoad, routePenalty,
        routeCustomers, deltaRemoval,
        pairU, pairV, numPairs,
        penaltyCapacity, penaltyDuration,
        outResults);
}
