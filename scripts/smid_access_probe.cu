// Report the physical SM IDs used by a CUDA kernel.
//
// Each CTA requests slightly more than half of an SM's shared memory, forcing
// at most one probe CTA to reside on an SM. Launching two waves then reveals
// the set of physical SMs to which the CUDA context can dispatch work. This is
// useful for validating CUDA MPS active-thread caps rather than merely trusting
// cudaDevAttrMultiProcessorCount.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <set>
#include <string>
#include <thread>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = (call);                                                 \
    if (error != cudaSuccess) {                                                 \
      std::cerr << #call << ": " << cudaGetErrorString(error) << "\n";         \
      return 1;                                                                \
    }                                                                          \
  } while (0)

__global__ void record_smid(unsigned int* block_smids, unsigned long long cycles) {
  extern __shared__ unsigned char shared[];
  if (threadIdx.x == 0) {
    unsigned int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    block_smids[blockIdx.x] = smid;
    unsigned long long start = clock64();
    while (clock64() - start < cycles) {
      asm volatile("" ::: "memory");
    }
    shared[0] = static_cast<unsigned char>(smid);
  }
  __syncthreads();
}

int main(int argc, char** argv) {
  int duration_ms = argc > 1 ? std::atoi(argv[1]) : 250;
  int start_delay_ms = argc > 2 ? std::atoi(argv[2]) : 0;
  if (duration_ms <= 0 || start_delay_ms < 0) {
    std::cerr << "usage: smid_access_probe [duration_ms] [start_delay_ms]\n";
    return 2;
  }

  CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  int shared_per_sm = 0;
  int shared_per_block = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(
      &shared_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0));
  CUDA_CHECK(cudaDeviceGetAttribute(
      &shared_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
  int dynamic_shared = std::min(shared_per_block, shared_per_sm / 2 + 256);
  dynamic_shared -= dynamic_shared % 256;
  if (dynamic_shared <= shared_per_sm / 2) {
    std::cerr << "cannot request enough shared memory for one CTA/SM: per_sm="
              << shared_per_sm << " per_block=" << shared_per_block << "\n";
    return 3;
  }
  CUDA_CHECK(cudaFuncSetAttribute(
      record_smid, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shared));

  int blocks = 2 * properties.multiProcessorCount;
  unsigned int* device_smids = nullptr;
  CUDA_CHECK(cudaMalloc(&device_smids, blocks * sizeof(unsigned int)));
  CUDA_CHECK(cudaMemset(device_smids, 0xff, blocks * sizeof(unsigned int)));
  if (start_delay_ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(start_delay_ms));
  }

  unsigned long long cycles =
      static_cast<unsigned long long>(properties.clockRate) * duration_ms;
  record_smid<<<blocks, 256, dynamic_shared>>>(device_smids, cycles);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<unsigned int> block_smids(blocks);
  CUDA_CHECK(cudaMemcpy(
      block_smids.data(), device_smids, blocks * sizeof(unsigned int),
      cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_smids));

  std::set<unsigned int> unique_smids(block_smids.begin(), block_smids.end());
  std::cout << "{\"device\":\"" << properties.name
            << "\",\"reported_sm_count\":" << properties.multiProcessorCount
            << ",\"unique_physical_sm_count\":" << unique_smids.size()
            << ",\"dynamic_shared_bytes\":" << dynamic_shared
            << ",\"block_smids\":[";
  for (std::size_t index = 0; index < block_smids.size(); ++index) {
    if (index) std::cout << ',';
    std::cout << block_smids[index];
  }
  std::cout << "],\"unique_physical_smids\":[";
  std::size_t index = 0;
  for (unsigned int smid : unique_smids) {
    if (index++) std::cout << ',';
    std::cout << smid;
  }
  std::cout << "]}\n";
  return 0;
}
