// GpuSwapStarKernel.cuh
// CUDA device functions and the SWAP* evaluation kernel.
// Only included by GpuLocalSearch.cu; not a public header.
#pragma once
#include <cuda_runtime.h>
#include "GpuDataLayout.cuh"

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

static constexpr double GPU_EPSILON = 1e-5;
static constexpr double GPU_INF     = 1e30;

__device__ inline double gpuPenExcLoad(double load, double cap, double pen)
{
    return fmax(0.0, load - cap) * pen;
}

__device__ inline double gpuPenExcDur(double dur, double lim, double pen)
{
    return fmax(0.0, dur - lim) * pen;
}

// Find the cheapest position to insert customer `custU` into the route described
// by routeCustomers[0..len-1], conceptually removing `custExclude` first
// (pass -1 for no exclusion).
//
// Returns:
//   bestCost — change in route *distance* (service times excluded).
//   bestPos  — insert U after this customer ID; 0 means "after the depot".
__device__ void findBestInsert(
    int custU,
    int custExclude,
    const int* __restrict__ routeCustomers,
    int len,
    const double* __restrict__ timeCost,
    int stride,
    double& bestCost,
    int&    bestPos)
{
    bestCost = GPU_INF;
    bestPos  = 0;       // default: insert after depot

    int prevCust = 0;   // depot index
    for (int k = 0; k <= len; k++)
    {
        // After all real customers, try the arc back to the depot.
        int currCust = (k < len) ? routeCustomers[k] : 0;

        if (currCust == custExclude)
        {
            // This customer is being removed — skip, prevCust stays as-is.
            continue;
        }

        // Cost of inserting custU on arc (prevCust -> currCust).
        double cost = timeCost[prevCust * stride + custU]
                    + timeCost[custU    * stride + currCust]
                    - timeCost[prevCust * stride + currCust];
        if (cost < bestCost)
        {
            bestCost = cost;
            bestPos  = prevCust;
        }
        prevCust = currCust;
    }
}

// ---------------------------------------------------------------------------
// Main SWAP* evaluation kernel
//
// Grid  : one block per route pair (blockIdx.x = pair index).
// Block : BLOCK_SIZE threads (tuned at launch site).
//
// Shared memory layout (dynamic, size computed by host):
//   [0                          ] int   sRouteU[lenU]
//   [lenU*4                     ] int   sRouteV[lenV]
//   [P1 (8-byte aligned)        ] double sDeltaU[lenU]
//   [P1 + lenU*8               ] double sDeltaV[lenV]
//   [P2 (8-byte aligned)        ] GpuSwapStarResult sReduce[BLOCK_SIZE]
// ---------------------------------------------------------------------------
__global__ void evalSwapStarKernel(
    const int*    __restrict__ routeStart,
    const int*    __restrict__ routeLen,
    const double* __restrict__ routeDuration,
    const double* __restrict__ routeLoad,
    const double* __restrict__ routePenalty,
    const int*    __restrict__ routeCustomers,
    const double* __restrict__ deltaRemoval,
    const double* __restrict__ timeCost,
    const double* __restrict__ demand,
    const double* __restrict__ service,
    int    nbClients,
    double vehicleCapacity,
    double durationLimit,
    double penaltyCapacity,
    double penaltyDuration,
    const int* __restrict__ pairU,
    const int* __restrict__ pairV,
    int    numPairs,
    GpuSwapStarResult* results)
{
    extern __shared__ char smem[];

    const int pairIdx = blockIdx.x;
    if (pairIdx >= numPairs) return;

    const int rU = pairU[pairIdx];
    const int rV = pairV[pairIdx];
    const int lenU   = routeLen[rU];
    const int lenV   = routeLen[rV];
    const int startU = routeStart[rU];
    const int startV = routeStart[rV];

    const double loadU = routeLoad[rU],     loadV = routeLoad[rV];
    const double durU  = routeDuration[rU], durV  = routeDuration[rV];
    const double penU  = routePenalty[rU],  penV  = routePenalty[rV];
    const int    stride = nbClients + 1;
    const int    tid    = threadIdx.x;

    // --- Shared memory pointers ---
    int* sRouteU = (int*)smem;
    int* sRouteV = sRouteU + lenU;

    // 8-byte align for doubles
    const size_t intsBytes = (size_t)(lenU + lenV) * sizeof(int);
    const size_t dblStart  = (intsBytes + 7u) & ~7u;
    double* sDeltaU = (double*)(smem + dblStart);
    double* sDeltaV = sDeltaU + lenU;

    // 8-byte align for GpuSwapStarResult array
    const size_t reducStart = (dblStart + (size_t)(lenU + lenV) * sizeof(double) + 7u) & ~7u;
    GpuSwapStarResult* sReduce = (GpuSwapStarResult*)(smem + reducStart);

    // --- Cooperative load into shared memory ---
    for (int i = tid; i < lenU; i += blockDim.x)
        sRouteU[i] = routeCustomers[startU + i];
    for (int i = tid; i < lenV; i += blockDim.x)
        sRouteV[i] = routeCustomers[startV + i];
    for (int i = tid; i < lenU; i += blockDim.x)
        sDeltaU[i] = deltaRemoval[startU + i];
    for (int i = tid; i < lenV; i += blockDim.x)
        sDeltaV[i] = deltaRemoval[startV + i];
    __syncthreads();

    // --- Local best for this thread ---
    GpuSwapStarResult local;
    local.moveCost = GPU_INF;
    local.routeU   = rU;
    local.routeV   = rV;
    local.nodeU    = -1;
    local.nodeV    = -1;
    local.bestPosU = 0;
    local.bestPosV = 0;
    local.moveType = 0;

    const int totalWork = lenU * lenV + lenU + lenV;

    for (int work = tid; work < totalWork; work += blockDim.x)
    {
        if (work < lenU * lenV)
        {
            // -------------------------------------------------------
            // SWAP: exchange sRouteU[i] with sRouteV[j]
            // -------------------------------------------------------
            const int i = work / lenV;
            const int j = work % lenV;
            const int custU_v = sRouteU[i];
            const int custV_v = sRouteV[j];

            const double demU  = demand[custU_v];
            const double demV  = demand[custV_v];
            const double dremU = sDeltaU[i];
            const double dremV = sDeltaV[j];
            const double svcU  = service[custU_v];
            const double svcV  = service[custV_v];

            // Capacity penalty deltas
            const double dpU = gpuPenExcLoad(loadU - demU + demV, vehicleCapacity, penaltyCapacity) - penU;
            const double dpV = gpuPenExcLoad(loadV - demV + demU, vehicleCapacity, penaltyCapacity) - penV;

            // Quick filter (mirrors CPU's early-exit in swapStar())
            if (dpU + dremU + dpV + dremV > GPU_EPSILON) continue;

            // Best insertion of custU_v into routeV \ custV_v
            double extraV; int posU;
            findBestInsert(custU_v, custV_v, sRouteV, lenV, timeCost, stride, extraV, posU);

            // Best insertion of custV_v into routeU \ custU_v
            double extraU; int posV;
            findBestInsert(custV_v, custU_v, sRouteU, lenU, timeCost, stride, extraU, posV);

            const double moveCost =
                dpU + dremU + extraU
              + dpV + dremV + extraV
              + gpuPenExcDur(durU + dremU + extraU + svcV - svcU, durationLimit, penaltyDuration)
              + gpuPenExcDur(durV + dremV + extraV - svcV + svcU, durationLimit, penaltyDuration);

            if (moveCost < local.moveCost)
            {
                local.moveCost = moveCost;
                local.nodeU    = custU_v;  local.nodeV    = custV_v;
                local.bestPosU = posU;     local.bestPosV = posV;
                local.moveType = 0;
            }
        }
        else if (work < lenU * lenV + lenU)
        {
            // -------------------------------------------------------
            // RELOCATE U -> routeV  (V stays in routeV)
            // -------------------------------------------------------
            const int i       = work - lenU * lenV;
            const int custU_v = sRouteU[i];

            const double demU  = demand[custU_v];
            const double dremU = sDeltaU[i];
            const double svcU  = service[custU_v];

            // Best insertion into full routeV (no exclusion)
            double extraV; int posU;
            findBestInsert(custU_v, -1, sRouteV, lenV, timeCost, stride, extraV, posU);

            const double moveCost =
                dremU + extraV
              + gpuPenExcLoad(loadU - demU, vehicleCapacity, penaltyCapacity) - penU
              + gpuPenExcLoad(loadV + demU, vehicleCapacity, penaltyCapacity) - penV
              + gpuPenExcDur(durU + dremU - svcU, durationLimit, penaltyDuration)
              + gpuPenExcDur(durV + extraV + svcU, durationLimit, penaltyDuration);

            if (moveCost < local.moveCost)
            {
                local.moveCost = moveCost;
                local.nodeU    = custU_v;  local.nodeV    = -1;
                local.bestPosU = posU;     local.bestPosV = -1;
                local.moveType = 1;
            }
        }
        else
        {
            // -------------------------------------------------------
            // RELOCATE V -> routeU  (U stays in routeU)
            // -------------------------------------------------------
            const int j       = work - lenU * lenV - lenU;
            const int custV_v = sRouteV[j];

            const double demV  = demand[custV_v];
            const double dremV = sDeltaV[j];
            const double svcV  = service[custV_v];

            // Best insertion into full routeU (no exclusion)
            double extraU; int posV;
            findBestInsert(custV_v, -1, sRouteU, lenU, timeCost, stride, extraU, posV);

            const double moveCost =
                dremV + extraU
              + gpuPenExcLoad(loadU + demV, vehicleCapacity, penaltyCapacity) - penU
              + gpuPenExcLoad(loadV - demV, vehicleCapacity, penaltyCapacity) - penV
              + gpuPenExcDur(durU + extraU + svcV, durationLimit, penaltyDuration)
              + gpuPenExcDur(durV + dremV - svcV, durationLimit, penaltyDuration);

            if (moveCost < local.moveCost)
            {
                local.moveCost = moveCost;
                local.nodeU    = -1;       local.nodeV    = custV_v;
                local.bestPosU = -1;       local.bestPosV = posV;
                local.moveType = 2;
            }
        }
    }

    // --- Block-level reduction: find best move across all threads ---
    sReduce[tid] = local;
    __syncthreads();

    if (tid == 0)
    {
        GpuSwapStarResult best = sReduce[0];
        for (int i = 1; i < blockDim.x; i++)
            if (sReduce[i].moveCost < best.moveCost)
                best = sReduce[i];
        results[pairIdx] = best;
    }
}
