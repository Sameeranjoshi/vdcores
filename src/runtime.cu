#include "dae2.cuh"
#include "runtime.cuh"

#include <hip/hip_runtime.h>
#include <iostream>
#include <array>

size_t set_smem_size(size_t smem_size) {
    hipError_t err = hipFuncSetAttribute(reinterpret_cast<const void*>(dae2),
        hipFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    );
    if (err != hipSuccess) {
        std::cerr << "Kernel set parameter failed: " << hipGetErrorString(err) << std::endl;
    }
    return smem_size;
}

hipError_t launch_dae(
  int numSMs,
  size_t smem_size,
  CInst* compute_instructions,
  MInst* memory_instructions,
  CUtensorMap* tma_descs,
  int * bars,
  uint64_t * profile,
  int64_t stream
) {
  // wait for all pre-launch meta-data copying
  hipDeviceSynchronize();
  hipStream_t cuda_stream = reinterpret_cast<hipStream_t>(stream);
  dae2<<<numSMs, numThreads, smem_size, cuda_stream>>>(
    compute_instructions,
    memory_instructions,
    tma_descs,
    bars,
    profile
  );
  // TODO(zhiyuang): check launch error here?

  hipDeviceSynchronize();

  return hipGetLastError();
}

#ifdef __HIP_PLATFORM_AMD__
// AMD STUB: TMA descriptors don't exist on CDNA3.
CUtensorMap create_tma_descriptor(
  CUtensorMapDataType,
  int,
  void *,
  std::array<uint64_t, 5>,
  std::array<uint32_t, 5>,
  CUtensorMapSwizzle,
  std::array<uint64_t, 5>
) {
  CUtensorMap desc{};
  std::cerr << "[VDCores][HIP] create_tma_descriptor called: TMA not supported on AMD\n";
  return desc;
}
#else
CUtensorMap create_tma_descriptor(
  CUtensorMapDataType data_type,
  int dims,
  void * base,
  std::array<uint64_t, 5> global_dims,
  std::array<uint32_t, 5> box_dims,
  CUtensorMapSwizzle swizzle,
  std::array<uint64_t, 5> global_strides_opt
) {
  assert(dims <= 5 && "Maximum 5 dimensions supported");

  CUtensorMap desc;

  int element_size = -1; // default to BF16

  if (data_type == CU_TENSOR_MAP_DATA_TYPE_UINT8) {
    element_size = 1;
  } else if (data_type == CU_TENSOR_MAP_DATA_TYPE_UINT16 ||
             data_type == CU_TENSOR_MAP_DATA_TYPE_BFLOAT16) {
    element_size = 2;
  } else if (data_type == CU_TENSOR_MAP_DATA_TYPE_UINT32 ||
             data_type == CU_TENSOR_MAP_DATA_TYPE_INT32) {
    element_size = 4;
  } else if (data_type == CU_TENSOR_MAP_DATA_TYPE_UINT64 ||
             data_type == CU_TENSOR_MAP_DATA_TYPE_INT64) {
    element_size = 8;
  }
  assert(element_size > 0 && "Unsupported data type");

  uint64_t global_strides[5];
  uint32_t box_strides[5];

  // Calculate global strides using cumulative products
  global_strides[0] = global_dims[0] * element_size;
  for (int i = 1; i < dims - 1; i++) {
    global_strides[i] = global_strides[i-1] * global_dims[i];
  }

  // Box strides are always 1 (contiguous within each tile)
  for (int i = 0; i < dims; i++) {
    box_strides[i] = 1;
  }

  auto result = cuTensorMapEncodeTiled(
    &desc,
    data_type,
    dims,
    base,
    global_dims.data(),
    // we go with a compact layout if no strides are provided
    global_strides_opt[0] == 0 ? global_strides : global_strides_opt.data(),
    box_dims.data(),
    box_strides,

    CU_TENSOR_MAP_INTERLEAVE_NONE,    // No interleaving
    swizzle,       // Swizzle mode
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B,  // No L2 promotion
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE // No special OOB handling
  );
  assert(result == hipSuccess && "Failed to create tensor map");

  return desc;
}
#endif  // __HIP_PLATFORM_AMD__
