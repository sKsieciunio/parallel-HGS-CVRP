#pragma once
#include <atomic>
#include <string>

class HardwareMonitor;

// Collected during a single HGS run. All fields are atomics so they can be
// safely accumulated from OpenMP worker threads (parallel offspring).
struct Telemetry
{
    // --- Local search (all invocations combined) ---
    std::atomic<long long> lsCallCount{0};
    std::atomic<long long> lsTotalNs{0};
    std::atomic<long long> riMovesApplied{0};

    // --- GPU SWAP* ---
    std::atomic<long long> gpuKernelCalls{0};
    std::atomic<long long> gpuPairsEvaluated{0};
    std::atomic<long long> gpuMovesApplied{0};
    std::atomic<long long> gpuSwapStarNs{0};

    // --- OpenMP SWAP* ---
    std::atomic<long long> ompKernelCalls{0};
    std::atomic<long long> ompPairsEvaluated{0};
    std::atomic<long long> ompMovesApplied{0};
    std::atomic<long long> ompSwapStarNs{0};

    // --- CPU SWAP* (sequential baseline) ---
    std::atomic<long long> cpuPairsEvaluated{0};
    std::atomic<long long> cpuMovesApplied{0};
    std::atomic<long long> cpuSwapStarNs{0};

    // --- Parallel offspring ---
    std::atomic<long long> offspringRounds{0};
    std::atomic<long long> offspringGenerated{0};
    std::atomic<long long> offspringNs{0};

    // --- Island model migrations ---
    std::atomic<long long> migrationsSent{0};
    std::atomic<long long> migrationsReceived{0};

    // --- Genetic algorithm ---
    std::atomic<long long> totalIterations{0};

    void printSummary(const std::string& instanceName, int seed, const std::string& mode,
                      double bestCost, double totalTimeSec) const;

    // hw is used to include GPU/CPU utilization columns; pass a stopped monitor.
    void exportCSV(const std::string& path,
                   const std::string& instanceName, int seed, const std::string& mode,
                   double bestCost, double totalTimeSec,
                   const HardwareMonitor& hw) const;
};
