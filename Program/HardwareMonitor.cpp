#include "HardwareMonitor.h"
#include <chrono>
#include <numeric>
#include <thread>
#include <algorithm>

#ifdef HAS_NVML
#  include <nvml.h>
#endif

// ── Platform-specific process CPU time ───────────────────────────────────────

#if defined(_WIN32)
#  define NOMINMAX
#  include <windows.h>

double HardwareMonitor::processUptimeCpuSec()
{
    FILETIME create, exit, kernel, user;
    if (!GetProcessTimes(GetCurrentProcess(), &create, &exit, &kernel, &user))
        return 0.0;
    // FILETIME is in 100-nanosecond intervals
    ULARGE_INTEGER k, u;
    k.LowPart = kernel.dwLowDateTime; k.HighPart = kernel.dwHighDateTime;
    u.LowPart = user.dwLowDateTime;   u.HighPart = user.dwHighDateTime;
    return static_cast<double>(k.QuadPart + u.QuadPart) / 1e7;
}

int HardwareMonitor::logicalCoreCount()
{
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return static_cast<int>(si.dwNumberOfProcessors);
}

#else  // POSIX (Linux, macOS)
#  include <sys/resource.h>
#  include <unistd.h>

double HardwareMonitor::processUptimeCpuSec()
{
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) return 0.0;
    return  usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1e6
          + usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1e6;
}

int HardwareMonitor::logicalCoreCount()
{
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? static_cast<int>(n) : 1;
}

#endif

// ── HardwareMonitor ──────────────────────────────────────────────────────────

HardwareMonitor::HardwareMonitor(int gpuIndex, int intervalMs)
    : gpuIndex_(gpuIndex), intervalMs_(intervalMs)
{
#ifdef HAS_NVML
    gpuAvail_ = (nvmlInit() == NVML_SUCCESS);
#endif
}

HardwareMonitor::~HardwareMonitor()
{
    if (running_.load()) stop();
#ifdef HAS_NVML
    if (gpuAvail_) nvmlShutdown();
#endif
}

void HardwareMonitor::start()
{
    running_.store(true);
    thread_ = std::thread(&HardwareMonitor::threadLoop, this);
}

void HardwareMonitor::stop()
{
    running_.store(false);
    if (thread_.joinable()) thread_.join();
}

void HardwareMonitor::threadLoop()
{
#ifdef HAS_NVML
    nvmlDevice_t device{};
    bool devOk = gpuAvail_ &&
                 (nvmlDeviceGetHandleByIndex(gpuIndex_, &device) == NVML_SUCCESS);
#endif

    int nCores = std::max(1, logicalCoreCount());

    using Clock = std::chrono::steady_clock;
    auto prevWall = Clock::now();
    double prevCpu = processUptimeCpuSec();

    while (running_.load(std::memory_order_relaxed))
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(intervalMs_));

        // ── GPU ──────────────────────────────────────────────────────────────
#ifdef HAS_NVML
        if (devOk)
        {
            nvmlUtilization_t util{};
            if (nvmlDeviceGetUtilizationRates(device, &util) == NVML_SUCCESS)
            {
                gpuCoreSamples_.push_back(util.gpu);
                gpuMemSamples_.push_back(util.memory);
            }
        }
#endif

        // ── CPU ──────────────────────────────────────────────────────────────
        auto  nowWall = Clock::now();
        double nowCpu = processUptimeCpuSec();

        double deltaCpu  = nowCpu - prevCpu;
        double deltaWall = std::chrono::duration<double>(nowWall - prevWall).count();

        if (deltaWall > 0.0)
        {
            double util = std::min(100.0, deltaCpu / deltaWall / nCores * 100.0);
            cpuUtilSamples_.push_back(util);
        }

        prevWall = nowWall;
        prevCpu  = nowCpu;
    }
}

double HardwareMonitor::avgGpuCoreUtil() const
{
    if (gpuCoreSamples_.empty()) return 0.0;
    return static_cast<double>(
               std::accumulate(gpuCoreSamples_.begin(), gpuCoreSamples_.end(), 0u))
           / gpuCoreSamples_.size();
}

double HardwareMonitor::avgGpuMemUtil() const
{
    if (gpuMemSamples_.empty()) return 0.0;
    return static_cast<double>(
               std::accumulate(gpuMemSamples_.begin(), gpuMemSamples_.end(), 0u))
           / gpuMemSamples_.size();
}

double HardwareMonitor::avgCpuUtil() const
{
    if (cpuUtilSamples_.empty()) return 0.0;
    return std::accumulate(cpuUtilSamples_.begin(), cpuUtilSamples_.end(), 0.0)
           / cpuUtilSamples_.size();
}

int HardwareMonitor::sampleCount() const
{
    return static_cast<int>(cpuUtilSamples_.size());
}
