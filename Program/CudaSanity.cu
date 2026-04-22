#include "CudaSanity.h"

#include <cuda_runtime.h>

#include <iostream>

namespace
{
    __global__ void squareKernel(const int *input, int *output)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx == 0)
        {
            output[0] = input[0] * input[0];
        }
    }
}

bool runCudaSanityCheck()
{
    constexpr int hostInput = 7;
    int hostOutput = 0;

    int *deviceInput = nullptr;
    int *deviceOutput = nullptr;

    cudaError_t status = cudaMalloc(&deviceInput, sizeof(int));
    if (status != cudaSuccess)
    {
        std::cout << "CUDA sanity check: cudaMalloc(input) failed: " << cudaGetErrorString(status) << std::endl;
        return false;
    }

    status = cudaMalloc(&deviceOutput, sizeof(int));
    if (status != cudaSuccess)
    {
        std::cout << "CUDA sanity check: cudaMalloc(output) failed: " << cudaGetErrorString(status) << std::endl;
        cudaFree(deviceInput);
        return false;
    }

    status = cudaMemcpy(deviceInput, &hostInput, sizeof(int), cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
    {
        std::cout << "CUDA sanity check: H2D copy failed: " << cudaGetErrorString(status) << std::endl;
        cudaFree(deviceInput);
        cudaFree(deviceOutput);
        return false;
    }

    squareKernel<<<1, 1>>>(deviceInput, deviceOutput);
    status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        std::cout << "CUDA sanity check: kernel launch failed: " << cudaGetErrorString(status) << std::endl;
        cudaFree(deviceInput);
        cudaFree(deviceOutput);
        return false;
    }

    status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
    {
        std::cout << "CUDA sanity check: kernel execution failed: " << cudaGetErrorString(status) << std::endl;
        cudaFree(deviceInput);
        cudaFree(deviceOutput);
        return false;
    }

    status = cudaMemcpy(&hostOutput, deviceOutput, sizeof(int), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
    {
        std::cout << "CUDA sanity check: D2H copy failed: " << cudaGetErrorString(status) << std::endl;
        cudaFree(deviceInput);
        cudaFree(deviceOutput);
        return false;
    }

    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    const bool ok = (hostOutput == 49);
    return ok;
}
