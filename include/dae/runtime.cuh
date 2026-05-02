#pragma once

#include "context.cuh"
#include <hip/hip_runtime.h>
#include <array>

// On AMD all CUtensorMap* / cuTensorMapEncodeTiled stubs live in hip_compat.cuh
// (which context.cuh pulls in).

// runtime interface for DAE kernels
size_t set_smem_size(size_t smem_size = (1024 * 212));

hipError_t launch_dae(
  int numSMs,
  size_t smem_size,
  CInst* compute_instructions,
  MInst* memory_instructions,
  CUtensorMap* tma_descs,
  int * bars,
  uint64_t * profile,
  int64_t stream
);

CUtensorMap create_tma_descriptor(
  CUtensorMapDataType data_type,
  int dims,
  void * base,
  std::array<uint64_t, 5> global_dims,
  std::array<uint32_t, 5> box_dims,
  CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE,
  std::array<uint64_t, 5> global_strides_opt = {0, 0, 0, 0, 0}
);

