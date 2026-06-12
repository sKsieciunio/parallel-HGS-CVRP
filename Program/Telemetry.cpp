#include "Telemetry.h"
#include "HardwareMonitor.h"
#include <fstream>
#include <iostream>
#include <iomanip>

void Telemetry::printSummary(const std::string& instanceName, int seed,
                              const std::string& mode,
                              double bestCost, double totalTimeSec) const
{
    long long lsCalls = lsCallCount.load();
    long long lsMs    = lsTotalNs.load() / 1'000'000;
    long long riMoves = riMovesApplied.load();
    long long ssNs    = gpuSwapStarNs.load() + ompSwapStarNs.load() + cpuSwapStarNs.load();
    long long ssMs    = ssNs / 1'000'000;
    long long riMs    = lsMs - ssMs;

    std::cout << "\n====== TELEMETRY SUMMARY ==============================\n";
    std::cout << "  instance : " << instanceName << "\n";
    std::cout << "  seed     : " << seed << "\n";
    std::cout << "  mode     : " << mode << "\n";
    std::cout << "  best     : " << std::fixed << std::setprecision(2) << bestCost << "\n";
    std::cout << "  iter     : " << totalIterations.load() << "\n";
    std::cout << "  time(s)  : " << std::fixed << std::setprecision(3) << totalTimeSec << "\n";

    std::cout << "\n  -- Local search --\n";
    std::cout << "  LS calls   : " << lsCalls << "\n";
    std::cout << "  LS time(ms): " << lsMs << "\n";
    std::cout << "  RI moves   : " << riMoves << "\n";
    std::cout << "  RI time(ms): " << riMs << "  (derived: LS - SWAP*)\n";
    std::cout << "  SS time(ms): " << ssMs << "\n";

    if (gpuKernelCalls.load() > 0)
    {
        long long calls   = gpuKernelCalls.load();
        long long pairs   = gpuPairsEvaluated.load();
        long long moved   = gpuMovesApplied.load();
        long long ms      = gpuSwapStarNs.load() / 1'000'000;
        double hitRate    = pairs > 0 ? 100.0 * moved / pairs : 0.0;
        std::cout << "\n  -- GPU SWAP* --\n";
        std::cout << "  kernel calls  : " << calls << "\n";
        std::cout << "  pairs eval    : " << pairs
                  << "  (avg/call: " << (calls > 0 ? pairs / calls : 0) << ")\n";
        std::cout << "  moves applied : " << moved
                  << "  (hit rate: " << std::fixed << std::setprecision(1) << hitRate << "%)\n";
        std::cout << "  time(ms)      : " << ms << "\n";
    }

    if (ompKernelCalls.load() > 0)
    {
        long long calls   = ompKernelCalls.load();
        long long pairs   = ompPairsEvaluated.load();
        long long moved   = ompMovesApplied.load();
        long long ms      = ompSwapStarNs.load() / 1'000'000;
        double hitRate    = pairs > 0 ? 100.0 * moved / pairs : 0.0;
        std::cout << "\n  -- OpenMP SWAP* --\n";
        std::cout << "  invocations   : " << calls << "\n";
        std::cout << "  pairs eval    : " << pairs
                  << "  (avg/call: " << (calls > 0 ? pairs / calls : 0) << ")\n";
        std::cout << "  moves applied : " << moved
                  << "  (hit rate: " << std::fixed << std::setprecision(1) << hitRate << "%)\n";
        std::cout << "  time(ms)      : " << ms << "\n";
    }

    if (cpuPairsEvaluated.load() > 0)
    {
        long long pairs   = cpuPairsEvaluated.load();
        long long moved   = cpuMovesApplied.load();
        long long ms      = cpuSwapStarNs.load() / 1'000'000;
        double hitRate    = pairs > 0 ? 100.0 * moved / pairs : 0.0;
        std::cout << "\n  -- CPU SWAP* --\n";
        std::cout << "  pairs eval    : " << pairs << "\n";
        std::cout << "  moves applied : " << moved
                  << "  (hit rate: " << std::fixed << std::setprecision(1) << hitRate << "%)\n";
        std::cout << "  time(ms)      : " << ms << "\n";
    }

    if (offspringRounds.load() > 0)
    {
        long long rounds = offspringRounds.load();
        long long gen    = offspringGenerated.load();
        long long ms     = offspringNs.load() / 1'000'000;
        std::cout << "\n  -- Parallel offspring --\n";
        std::cout << "  rounds      : " << rounds << "\n";
        std::cout << "  generated   : " << gen << "\n";
        std::cout << "  time(ms)    : " << ms << "\n";
    }

    if (migrationsSent.load() > 0 || migrationsReceived.load() > 0)
    {
        std::cout << "\n  -- Island migrations --\n";
        std::cout << "  sent     : " << migrationsSent.load() << "\n";
        std::cout << "  received : " << migrationsReceived.load() << "\n";
    }

    std::cout << "=======================================================\n\n";
}

void Telemetry::exportCSV(const std::string& path,
                           const std::string& instanceName, int seed,
                           const std::string& mode,
                           double bestCost, double totalTimeSec,
                           const HardwareMonitor& hw) const
{
    bool writeHeader = false;
    {
        std::ifstream check(path);
        writeHeader = !check.good() || check.peek() == std::ifstream::traits_type::eof();
    }

    std::ofstream out(path, std::ios::app);
    if (!out.is_open()) return;

    if (writeHeader)
    {
        out << "instance;seed;mode;bestCost;totalIter;totalTimeSec"
               ";lsCallCount;lsTotalMs;riMovesApplied"
               ";gpuKernelCalls;gpuPairsEvaluated;gpuMovesApplied;gpuSwapStarMs"
               ";ompKernelCalls;ompPairsEvaluated;ompMovesApplied;ompSwapStarMs"
               ";cpuPairsEvaluated;cpuMovesApplied;cpuSwapStarMs"
               ";offspringRounds;offspringGenerated;offspringMs"
               ";migrationsSent;migrationsReceived"
               ";hwSamples;avgCpuUtil;avgGpuCoreUtil;avgGpuMemUtil\n";
    }

    out << instanceName << ";" << seed << ";" << mode << ";"
        << std::fixed << std::setprecision(2) << bestCost << ";"
        << totalIterations.load() << ";"
        << std::setprecision(3) << totalTimeSec << ";"
        << lsCallCount.load() << ";"
        << lsTotalNs.load() / 1'000'000 << ";"
        << riMovesApplied.load() << ";"
        << gpuKernelCalls.load() << ";"
        << gpuPairsEvaluated.load() << ";"
        << gpuMovesApplied.load() << ";"
        << gpuSwapStarNs.load() / 1'000'000 << ";"
        << ompKernelCalls.load() << ";"
        << ompPairsEvaluated.load() << ";"
        << ompMovesApplied.load() << ";"
        << ompSwapStarNs.load() / 1'000'000 << ";"
        << cpuPairsEvaluated.load() << ";"
        << cpuMovesApplied.load() << ";"
        << cpuSwapStarNs.load() / 1'000'000 << ";"
        << offspringRounds.load() << ";"
        << offspringGenerated.load() << ";"
        << offspringNs.load() / 1'000'000 << ";"
        << migrationsSent.load() << ";"
        << migrationsReceived.load() << ";"
        << hw.sampleCount() << ";"
        << std::setprecision(1) << hw.avgCpuUtil() << ";"
        << hw.avgGpuCoreUtil() << ";"
        << hw.avgGpuMemUtil() << "\n";
}
