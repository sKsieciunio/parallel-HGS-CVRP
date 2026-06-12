#pragma once
#include <thread>
#include <atomic>
#include <vector>
#include <cstdint>

// Samples GPU and CPU utilization in a background thread.
// GPU sampling requires NVML (compiled in when HAS_NVML is defined).
// CPU sampling uses OS process APIs (GetProcessTimes / getrusage).
class HardwareMonitor
{
public:
    // gpuIndex  : which GPU to monitor (0 = first)
    // intervalMs: sampling interval in milliseconds
    explicit HardwareMonitor(int gpuIndex = 0, int intervalMs = 500);
    ~HardwareMonitor();

    void start();
    void stop();   // blocks until the sampling thread exits

    // Results (valid after stop())
    double avgGpuCoreUtil() const;   // 0–100 %
    double avgGpuMemUtil()  const;   // 0–100 %
    double avgCpuUtil()     const;   // 0–100 % across all logical cores
    int    sampleCount()    const;
    bool   gpuMonEnabled()  const { return gpuAvail_; }

private:
    int  gpuIndex_;
    int  intervalMs_;
    bool gpuAvail_ = false;

    std::atomic<bool> running_{ false };
    std::thread       thread_;

    std::vector<unsigned int> gpuCoreSamples_;
    std::vector<unsigned int> gpuMemSamples_;

    // CPU: we take a snapshot of process CPU time + wall time at each tick
    // and compute utilization as delta-CPU / (delta-wall * nCores)
    std::vector<double> cpuUtilSamples_;

    void threadLoop();

    // Returns accumulated process CPU seconds (user + kernel, all threads).
    static double processUptimeCpuSec();
    static int    logicalCoreCount();
};
