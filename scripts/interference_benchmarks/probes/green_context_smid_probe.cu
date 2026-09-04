// Validate CUDA Green Context SM isolation with physical %smid observations.
//
// The probe splits one device's complete SM resource into two partitions,
// creates a Green Context for each partition, and runs a one-CTA-per-SM kernel
// in each context.  Disjoint physical SM-ID sets demonstrate spatial
// isolation; device properties alone are not sufficient evidence.

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <set>
#include <vector>

#define CU_CHECK(call)                                                        \
  do {                                                                        \
    CUresult result = (call);                                                 \
    if (result != CUDA_SUCCESS) {                                             \
      const char* name = nullptr;                                             \
      const char* message = nullptr;                                          \
      cuGetErrorName(result, &name);                                          \
      cuGetErrorString(result, &message);                                     \
      std::cerr << #call << ": " << (name ? name : "unknown") << " ("      \
                << (message ? message : "unknown") << ")\n";                \
      return 1;                                                               \
    }                                                                         \
  } while (0)

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t error = (call);                                               \
    if (error != cudaSuccess) {                                               \
      std::cerr << #call << ": " << cudaGetErrorString(error) << "\n";       \
      return {};                                                              \
    }                                                                         \
  } while (0)

__global__ void record_smid(unsigned int* block_smids,
                            unsigned long long cycles) {
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

std::set<unsigned int> observe_smids(CUcontext context, unsigned int sm_count,
                                     int duration_ms) {
  if (cuCtxSetCurrent(context) != CUDA_SUCCESS) {
    std::cerr << "cuCtxSetCurrent failed\n";
    return {};
  }

  int shared_per_sm = 0;
  int shared_per_block = 0;
  int clock_rate_khz = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(
      &shared_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0));
  CUDA_CHECK(cudaDeviceGetAttribute(
      &shared_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
  CUDA_CHECK(cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, 0));
  int dynamic_shared = std::min(shared_per_block, shared_per_sm / 2 + 256);
  dynamic_shared -= dynamic_shared % 256;
  if (dynamic_shared <= shared_per_sm / 2) {
    std::cerr << "unable to limit residency to one CTA per SM\n";
    return {};
  }
  CUDA_CHECK(cudaFuncSetAttribute(
      record_smid, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_shared));

  const unsigned int blocks = std::max(4U * sm_count, 32U);
  unsigned int* device_smids = nullptr;
  CUDA_CHECK(cudaMalloc(&device_smids, blocks * sizeof(unsigned int)));
  CUDA_CHECK(cudaMemset(device_smids, 0xff, blocks * sizeof(unsigned int)));
  const unsigned long long cycles =
      static_cast<unsigned long long>(clock_rate_khz) * duration_ms;
  record_smid<<<blocks, 256, dynamic_shared>>>(device_smids, cycles);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<unsigned int> block_smids(blocks);
  CUDA_CHECK(cudaMemcpy(block_smids.data(), device_smids,
                        blocks * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_smids));
  return {block_smids.begin(), block_smids.end()};
}

void print_ids(const std::set<unsigned int>& ids) {
  std::cout << '[';
  std::size_t index = 0;
  for (unsigned int id : ids) {
    if (index++) std::cout << ',';
    std::cout << id;
  }
  std::cout << ']';
}

int main(int argc, char** argv) {
  const unsigned int requested_sms =
      argc > 1 ? static_cast<unsigned int>(std::atoi(argv[1])) : 8U;
  const int duration_ms = argc > 2 ? std::atoi(argv[2]) : 10;
  if (requested_sms == 0 || duration_ms <= 0) {
    std::cerr << "usage: green_context_smid_probe [partition_sms] [duration_ms]\n";
    return 2;
  }

  CU_CHECK(cuInit(0));
  CUdevice device = 0;
  CU_CHECK(cuDeviceGet(&device, 0));
  CUdevResource complete{};
  CUdevResource partition{};
  CUdevResource remaining{};
  CU_CHECK(cuDeviceGetDevResource(device, &complete, CU_DEV_RESOURCE_TYPE_SM));

  unsigned int group_count = 1;
  CU_CHECK(cuDevSmResourceSplitByCount(&partition, &group_count, &complete,
                                       &remaining, 0, requested_sms));
  if (group_count != 1 || partition.sm.smCount == 0 ||
      remaining.sm.smCount == 0) {
    std::cerr << "split did not produce two nonempty SM resources\n";
    return 3;
  }

  CUdevResourceDesc partition_desc = nullptr;
  CUdevResourceDesc remaining_desc = nullptr;
  CU_CHECK(cuDevResourceGenerateDesc(&partition_desc, &partition, 1));
  CU_CHECK(cuDevResourceGenerateDesc(&remaining_desc, &remaining, 1));
  CUgreenCtx partition_green = nullptr;
  CUgreenCtx remaining_green = nullptr;
  CU_CHECK(cuGreenCtxCreate(&partition_green, partition_desc, device,
                            CU_GREEN_CTX_DEFAULT_STREAM));
  CU_CHECK(cuGreenCtxCreate(&remaining_green, remaining_desc, device,
                            CU_GREEN_CTX_DEFAULT_STREAM));
  CUcontext partition_context = nullptr;
  CUcontext remaining_context = nullptr;
  CU_CHECK(cuCtxFromGreenCtx(&partition_context, partition_green));
  CU_CHECK(cuCtxFromGreenCtx(&remaining_context, remaining_green));

  const std::set<unsigned int> partition_ids =
      observe_smids(partition_context, partition.sm.smCount, duration_ms);
  const std::set<unsigned int> remaining_ids =
      observe_smids(remaining_context, remaining.sm.smCount, duration_ms);
  CU_CHECK(cuCtxSetCurrent(nullptr));

  std::vector<unsigned int> intersection;
  std::set_intersection(partition_ids.begin(), partition_ids.end(),
                        remaining_ids.begin(), remaining_ids.end(),
                        std::back_inserter(intersection));
  std::cout << "{\"device_total_sms\":" << complete.sm.smCount
            << ",\"requested_partition_sms\":" << requested_sms
            << ",\"actual_partition_sms\":" << partition.sm.smCount
            << ",\"remaining_partition_sms\":" << remaining.sm.smCount
            << ",\"first_observed_sm_count\":" << partition_ids.size()
            << ",\"second_observed_sm_count\":" << remaining_ids.size()
            << ",\"intersection_count\":" << intersection.size()
            << ",\"first_physical_smids\":";
  print_ids(partition_ids);
  std::cout << ",\"second_physical_smids\":";
  print_ids(remaining_ids);
  std::cout << "}\n";

  CU_CHECK(cuGreenCtxDestroy(partition_green));
  CU_CHECK(cuGreenCtxDestroy(remaining_green));
  return intersection.empty() && partition_ids.size() == partition.sm.smCount &&
                 remaining_ids.size() == remaining.sm.smCount
             ? 0
             : 4;
}
