// GpuDataLayout.cuh
// Plain C++ compatible data structures for GPU<->CPU communication.
// No CUDA types — safe to include from regular .cpp translation units.
#pragma once

// Result describing the best SWAP* move found for one evaluated route pair.
// Stored as plain POD so it can live in shared memory and be shuffled between
// host and device without any special handling.
struct GpuSwapStarResult
{
    double moveCost;   // Total cost delta; 1e30 = no improving move found.
    int    nodeU;      // Customer moved from routeU  (-1 = none, relocate-V only).
    int    nodeV;      // Customer moved from routeV  (-1 = none, relocate-U only).
    int    bestPosU;   // Insert U after this customer ID in routeV (0 = after depot).
    int    bestPosV;   // Insert V after this customer ID in routeU (0 = after depot).
    int    routeU;     // Source route index for nodeU.
    int    routeV;     // Source route index for nodeV.
    int    moveType;   // 0 = swap U<->V, 1 = relocate U->routeV, 2 = relocate V->routeU.
};
