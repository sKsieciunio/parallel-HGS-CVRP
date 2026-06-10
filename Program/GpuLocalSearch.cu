// GpuLocalSearch.cu
// Host-side orchestration: device memory management, data transfer, kernel
// dispatch, and result selection for GPU-accelerated SWAP* evaluation.
#include "GpuLocalSearch.h"
#include "GpuSwapStarKernel.cuh"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

static size_t align8(size_t x) { return (x + 7u) & ~7u; }

// ---------------------------------------------------------------------------
// Error checking helpers
// ---------------------------------------------------------------------------
#define CUDA_LOG(call)                                                  \
	do {                                                                \
		cudaError_t _err = (call);                                      \
		if (_err != cudaSuccess)                                        \
			std::fprintf(stderr, "CUDA error %s:%d - %s\n",             \
						 __FILE__, __LINE__, cudaGetErrorString(_err)); \
	} while (0)

#define CUDA_RET_NULL(call)                                             \
	do {                                                                \
		cudaError_t _err = (call);                                      \
		if (_err != cudaSuccess) {                                      \
			std::fprintf(stderr, "CUDA error %s:%d - %s\n",             \
						 __FILE__, __LINE__, cudaGetErrorString(_err)); \
			return nullptr;                                             \
		}                                                               \
	} while (0)

#define CUDA_RET_FALSE(call)                                            \
	do {                                                                \
		cudaError_t _err = (call);                                      \
		if (_err != cudaSuccess) {                                      \
			std::fprintf(stderr, "CUDA error %s:%d - %s\n",             \
						 __FILE__, __LINE__, cudaGetErrorString(_err)); \
			return false;                                               \
		}                                                               \
	} while (0)

// ---------------------------------------------------------------------------
// Block size for SWAP* kernel (tunable)
// ---------------------------------------------------------------------------
static constexpr int BLOCK_SIZE = 256;

// ---------------------------------------------------------------------------
// Shared, read-only problem data (uploaded once, referenced by every instance).
// One per process (or per island/rank). Read-only on the device.
// ---------------------------------------------------------------------------
struct GpuProblemData
{
	double* d_timeCost	= nullptr; // (nbClients+1)^2  row-major
	double* d_demand	= nullptr; // nbClients+1
	double* d_service	= nullptr; // nbClients+1
	int nbClients = 0;
};

GpuProblemData * createGpuProblemData(
	const double * timeCostFlat,
	const double * demand,
	const double * service,
	int nbClients)
{
	int deviceCount = 0;
	if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
	{
		std::fprintf(stderr, "GpuProblemData: no CUDA device available.\n");
		return nullptr;
	}

	GpuProblemData* pd = new GpuProblemData();
	pd->nbClients = nbClients;
	const int n = nbClients + 1;

	CUDA_RET_NULL(cudaMalloc(&pd->d_timeCost, (size_t)n * n * sizeof(double)));
	CUDA_RET_NULL(cudaMalloc(&pd->d_demand, (size_t)n * sizeof(double)));
	CUDA_RET_NULL(cudaMalloc(&pd->d_service, (size_t)n * sizeof(double)));

	CUDA_RET_NULL(cudaMemcpy(pd->d_timeCost, timeCostFlat,
							 (size_t)n * n * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_RET_NULL(cudaMemcpy(pd->d_demand, demand,
							 (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_RET_NULL(cudaMemcpy(pd->d_service, service,
							 (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
	return pd;
}

void destroyGpuProblemData(GpuProblemData * pd)
{
	if (!pd) return;
	cudaFree(pd->d_timeCost);
	cudaFree(pd->d_demand);
	cudaFree(pd->d_service);
	delete pd;
}

// ---------------------------------------------------------------------------
// GpuLocalSearch class — per-instance state (one per host thread).
// ---------------------------------------------------------------------------
class GpuLocalSearch
{
  public:
	int		nbClients_ = 0;
	int		nbVehicles_ = 0;
	int		maxPairs_ = 0;
	double	vehicleCapacity_ = 0.0;
	double	durationLimit_ = 0.0;

	GpuProblemData * problem_ = nullptr; // shared, read-only (not owned unless ownsProblem_)
	bool ownsProblem_ = false;

	cudaStream_t stream_ = nullptr;

	// ---- Single flat buffer for all per-call route + pair data ----
	char*	h_routeFlat_ = nullptr; // pinned host staging buffer
	char*	d_routeFlat_ = nullptr; // device mirror
	size_t	flatSize_ = 0;			// total bytes

	// Byte offsets into the flat buffer for each logical array
	size_t off_routeDuration_;
	size_t off_routeLoad_;
	size_t off_routePenalty_;
	size_t off_deltaRemoval_;
	size_t off_routeStart_;
	size_t off_routeLen_;
	size_t off_routeCustomers_;
	size_t off_pairU_;
	size_t off_pairV_;

	// Device sub-pointers (point into d_routeFlat_; set once at init)
	double* d_routeDuration_ = nullptr;
	double * d_routeLoad_ = nullptr;
	double * d_routePenalty_ = nullptr;
	double * d_deltaRemoval_ = nullptr;
	int * d_routeStart_ = nullptr;
	int * d_routeLen_ = nullptr;
	int * d_routeCustomers_ = nullptr;
	int * d_pairU_ = nullptr;
	int * d_pairV_ = nullptr;

	GpuSwapStarResult* d_results_ = nullptr;  // maxPairs — separate allocation

	GpuLocalSearch() = default;

	bool evaluateSwapStar(
		const int * routeStart,
		const int * routeLen,
		const double * routeDuration,
		const double * routeLoad,
		const double * routePenalty,
		const int * routeCustomers,
		const double * deltaRemoval,
		const int * pairU,
		const int * pairV,
		int numPairs,
		double penaltyCapacity,
		double penaltyDuration,
		GpuSwapStarResult * outResults)
	{
		if (numPairs == 0) return false;

		// Pack all per-call arrays into the pinned host staging buffer,
		// then upload with a single cudaMemcpy instead of 9 separate ones.
        std::memcpy(h_routeFlat_ + off_routeDuration_,  routeDuration,  nbVehicles_ * sizeof(double));
        std::memcpy(h_routeFlat_ + off_routeLoad_,      routeLoad,      nbVehicles_ * sizeof(double));
        std::memcpy(h_routeFlat_ + off_routePenalty_,   routePenalty,   nbVehicles_ * sizeof(double));
        std::memcpy(h_routeFlat_ + off_deltaRemoval_,   deltaRemoval,   nbClients_  * sizeof(double));
        std::memcpy(h_routeFlat_ + off_routeStart_,     routeStart,     nbVehicles_ * sizeof(int));
        std::memcpy(h_routeFlat_ + off_routeLen_,       routeLen,       nbVehicles_ * sizeof(int));
        std::memcpy(h_routeFlat_ + off_routeCustomers_, routeCustomers, nbClients_  * sizeof(int));
        std::memcpy(h_routeFlat_ + off_pairU_,          pairU,          numPairs    * sizeof(int));
        std::memcpy(h_routeFlat_ + off_pairV_,          pairV,          numPairs    * sizeof(int));

		CUDA_RET_FALSE(cudaMemcpyAsync(d_routeFlat_, h_routeFlat_, off_pairU_,
									   cudaMemcpyHostToDevice, stream_));
		CUDA_RET_FALSE(cudaMemcpyAsync(d_routeFlat_ + off_pairU_, h_routeFlat_ + off_pairU_,
									   (size_t)numPairs * sizeof(int),
									   cudaMemcpyHostToDevice, stream_));
		CUDA_RET_FALSE(cudaMemcpyAsync(d_routeFlat_ + off_pairV_, h_routeFlat_ + off_pairV_,
									   (size_t)numPairs * sizeof(int),
									   cudaMemcpyHostToDevice, stream_));

		// Shared memory layout (see GpuSwapStarKernel.cuh for details):
		//   int[maxLen*2] + double[maxLen*2] + GpuSwapStarResult[BLOCK_SIZE]
		// Each section is 8-byte aligned.
		const int maxLen = *std::max_element(routeLen, routeLen + nbVehicles_);
		const size_t intsBytes = ((size_t)(2 * maxLen) * sizeof(int) + 7u) & ~7u;
		const size_t dblsBytes = (size_t)(2 * maxLen) * sizeof(double);
		const size_t reducBytes = ((intsBytes + dblsBytes + 7u) & ~7u) - (intsBytes + dblsBytes) + (size_t)(BLOCK_SIZE / 32) * sizeof(GpuSwapStarResult);
		const size_t smemSize = intsBytes + dblsBytes + reducBytes;

		// One block per route pair — all pairs evaluated in parallel
		evalSwapStarKernel<<<numPairs, BLOCK_SIZE, smemSize, stream_>>>(
			d_routeStart_, d_routeLen_, d_routeDuration_, d_routeLoad_, d_routePenalty_,
			d_routeCustomers_, d_deltaRemoval_,
			problem_->d_timeCost, problem_->d_demand, problem_->d_service,
			nbClients_, vehicleCapacity_, durationLimit_,
			penaltyCapacity, penaltyDuration,
			d_pairU_, d_pairV_, numPairs,
			d_results_);

		CUDA_RET_FALSE(cudaGetLastError());

		CUDA_RET_FALSE(cudaMemcpyAsync(outResults, d_results_,
									   (size_t)numPairs * sizeof(GpuSwapStarResult),
									   cudaMemcpyDeviceToHost, stream_));

		CUDA_RET_FALSE(cudaStreamSynchronize(stream_));

		for (int i = 0; i < numPairs; i++)
			if (outResults[i].moveCost < -1e-5)
				return true;
		return false;
	}
};

// ---------------------------------------------------------------------------
// Per-instance allocation given a shared problem context.
// ---------------------------------------------------------------------------
GpuLocalSearch * createGpuLocalSearchShared(
	GpuProblemData * shared,
	int nbClients,
	int nbVehicles,
	double vehicleCapacity,
	double durationLimit)
{
	if (!shared) return nullptr;

	GpuLocalSearch * ls = new GpuLocalSearch();
	ls->problem_ = shared;
	ls->ownsProblem_ = false;
	ls->nbClients_ = nbClients;
	ls->nbVehicles_ = nbVehicles;
	ls->vehicleCapacity_ = vehicleCapacity;
	ls->durationLimit_ = durationLimit;

	const int maxPairs = nbVehicles * (nbVehicles - 1) / 2;
	ls->maxPairs_ = maxPairs;

	// Flat buffer layout (doubles first for natural alignment).
	size_t off = 0;
	ls->off_routeDuration_ = off;
	off += align8((size_t)nbVehicles * sizeof(double));
	ls->off_routeLoad_ = off;
	off += align8((size_t)nbVehicles * sizeof(double));
	ls->off_routePenalty_ = off;
	off += align8((size_t)nbVehicles * sizeof(double));
	ls->off_deltaRemoval_ = off;
	off += align8((size_t)nbClients * sizeof(double));
	ls->off_routeStart_ = off;
	off += align8((size_t)nbVehicles * sizeof(int));
	ls->off_routeLen_ = off;
	off += align8((size_t)nbVehicles * sizeof(int));
	ls->off_routeCustomers_ = off;
	off += align8((size_t)nbClients * sizeof(int));
	ls->off_pairU_ = off;
	off += align8((size_t)maxPairs * sizeof(int));
	ls->off_pairV_ = off;
	off += align8((size_t)maxPairs * sizeof(int));
	ls->flatSize_ = off;

	// Pinned host staging buffer + matching device buffer
	CUDA_RET_NULL(cudaMallocHost(&ls->h_routeFlat_, ls->flatSize_));
	CUDA_RET_NULL(cudaMalloc(&ls->d_routeFlat_, ls->flatSize_));

    // Set device sub-pointers once; they remain valid for the object lifetime
    ls->d_routeDuration_  = (double*)(ls->d_routeFlat_ + ls->off_routeDuration_);
    ls->d_routeLoad_      = (double*)(ls->d_routeFlat_ + ls->off_routeLoad_);
    ls->d_routePenalty_   = (double*)(ls->d_routeFlat_ + ls->off_routePenalty_);
    ls->d_deltaRemoval_   = (double*)(ls->d_routeFlat_ + ls->off_deltaRemoval_);
    ls->d_routeStart_     = (int*)   (ls->d_routeFlat_ + ls->off_routeStart_);
    ls->d_routeLen_       = (int*)   (ls->d_routeFlat_ + ls->off_routeLen_);
    ls->d_routeCustomers_ = (int*)   (ls->d_routeFlat_ + ls->off_routeCustomers_);
    ls->d_pairU_          = (int*)   (ls->d_routeFlat_ + ls->off_pairU_);
    ls->d_pairV_          = (int*)   (ls->d_routeFlat_ + ls->off_pairV_);

	CUDA_RET_NULL(cudaMalloc(&ls->d_results_, (size_t)maxPairs * sizeof(GpuSwapStarResult)));
	CUDA_RET_NULL(cudaStreamCreate(&ls->stream_));
	return ls;
}

GpuLocalSearch * createGpuLocalSearch(
	const double * timeCostFlat,
	const double * demand,
	const double * service,
	int nbClients,
	int nbVehicles,
	double vehicleCapacity,
	double durationLimit)
{
	GpuProblemData * pd = createGpuProblemData(timeCostFlat, demand, service, nbClients);
	if (!pd) return nullptr;

	GpuLocalSearch * ls = createGpuLocalSearchShared(pd, nbClients, nbVehicles,
													 vehicleCapacity, durationLimit);
	if (!ls) {
		destroyGpuProblemData(pd);
		return nullptr;
	}

	ls->ownsProblem_ = true;
	return ls;
}

void destroyGpuLocalSearch(GpuLocalSearch* ls)
{
	if (!ls) return;
	if (ls->stream_) cudaStreamDestroy(ls->stream_);
	cudaFreeHost(ls->h_routeFlat_);
	cudaFree(ls->d_routeFlat_);
	cudaFree(ls->d_results_);
	if (ls->ownsProblem_) destroyGpuProblemData(ls->problem_);
	delete ls;
}

bool gpuEvaluateSwapStar(
	GpuLocalSearch * handle,
	const int * routeStart,
	const int * routeLen,
	const double * routeDuration,
	const double * routeLoad,
	const double * routePenalty,
	const int * routeCustomers,
	const double * deltaRemoval,
	const int * pairU,
	const int * pairV,
	int numPairs,
	double penaltyCapacity,
	double penaltyDuration,
	GpuSwapStarResult * outResults)
{
	return handle->evaluateSwapStar(
		routeStart, routeLen, routeDuration, routeLoad, routePenalty,
		routeCustomers, deltaRemoval,
		pairU, pairV, numPairs,
		penaltyCapacity, penaltyDuration,
		outResults);
}
