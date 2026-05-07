#pragma once
//
// HIP compatibility shim for VDCores.
//
// On NVIDIA: just re-include the real <cuda/barrier> and <cuda/ptx>.
// On AMD HIP: provide stand-in cuda::barrier / cuda::ptx::* / cuda::device::*
// declarations so the rest of the codebase compiles. The semantics are
// approximate — TMA paths are stubbed, WGMMA paths are stubbed, but the
// barrier and sreg helpers are functional enough to run non-Hopper-specific
// kernels (RMSNorm, SiLU, Argmax, vector ops).
//
// What still does NOT work on AMD after this header:
//   * cp_async_bulk*       -> stubbed (calls __builtin_trap on use)
//   * fence_proxy_async    -> __threadfence
//   * WGMMA tasks          -> see include/task/wgmma.cuh, stubbed there
//
// See PORT_STATUS.md for the bigger picture.
//

#if !defined(__HIP_PLATFORM_AMD__)

// ===== NVIDIA / CUDA path: real headers =====
#include <cuda/barrier>
#include <cuda/ptx>

#else  // __HIP_PLATFORM_AMD__

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
// hip_bf16.h (note the abbreviated name) provides __hip_bfloat16 / __hip_bfloat162
// as full struct types, matching the CUDA-side leading-underscore convention.
// Including it here makes those names available in every compile that pulls in
// hip_compat.cuh, instead of relying on torch headers to drag it in transitively.
#include <hip/hip_bf16.h>
#include <cstdint>

// CUDA warp-sync intrinsics -> HIP non-sync equivalents.
// (HIP wavefronts are implicitly synchronous within a wave.)
#ifndef __shfl_sync
#define __shfl_sync(mask, val, lane, ...)        __shfl((val), (lane), ##__VA_ARGS__)
#endif
#ifndef __shfl_up_sync
#define __shfl_up_sync(mask, val, delta, ...)    __shfl_up((val), (delta), ##__VA_ARGS__)
#endif
#ifndef __shfl_down_sync
#define __shfl_down_sync(mask, val, delta, ...)  __shfl_down((val), (delta), ##__VA_ARGS__)
#endif
#ifndef __shfl_xor_sync
#define __shfl_xor_sync(mask, val, lane, ...)    __shfl_xor((val), (lane), ##__VA_ARGS__)
#endif
#ifndef __ballot_sync
#define __ballot_sync(mask, pred)                __ballot((pred))
#endif
#ifndef __any_sync
#define __any_sync(mask, pred)                   __any((pred))
#endif
#ifndef __all_sync
#define __all_sync(mask, pred)                   __all((pred))
#endif
#ifndef __activemask
#define __activemask()                           __ballot(1)
#endif
#ifndef __syncwarp
#define __syncwarp(...)                          ((void)0)
#endif

// cutlass::bfloat16_t -> hip_bfloat16 alias so headers that name the cutlass
// type still typecheck. We don't pull in cutlass on AMD.
namespace cutlass {
using bfloat16_t = hip_bfloat16;
}

// __nanosleep -> AMD s_sleep. Approximate; s_sleep takes 1..127 in 64-cycle
// units, but for the spin-poll backoff this is fine.
__device__ __forceinline__ void __nanosleep(unsigned ns) {
  __builtin_amdgcn_s_sleep(1);
  (void)ns;
}

// make_bfloat162 — CUDA helper not present on AMD. Build the native
// __hip_bfloat162 from two halves. (ROCm 7.x already provides __hip_bfloat16
// and __hip_bfloat162 as full struct types via amd_hip_bf16.h, so we do NOT
// re-typedef them here.)
__device__ __forceinline__ __hip_bfloat162 make_bfloat162(__hip_bfloat16 a, __hip_bfloat16 b) {
  __hip_bfloat162 r;
  r.x = a;
  r.y = b;
  return r;
}

// Minimal cute:: namespace stubs. Just enough to let dispatch.cuh declare
// `using gemm_atom = cute::SM90_*<...>` and pass it as a template param into
// task_gemm — which on AMD is itself a stub that traps. None of these types
// have to *do* anything, they just need to exist for name lookup.
namespace cute {
namespace GMMA {
  enum Major { K, MN };
  template <typename T> struct Layout_K_SW128_Atom {};
  template <typename T> struct Layout_MN_SW128_Atom {};
}
template <int N> struct Int { static constexpr int value = N; };
template <GMMA::Major A = GMMA::K, GMMA::Major B = GMMA::K> struct SM90_64x64x16_F32BF16BF16_SS {};
template <GMMA::Major A = GMMA::K, GMMA::Major B = GMMA::K> struct SM90_64x64x16_F32BF16BF16_RS {};
template <GMMA::Major A = GMMA::K, GMMA::Major B = GMMA::K> struct SM90_64x128x16_F32BF16BF16_SS {};
template <GMMA::Major A = GMMA::K, GMMA::Major B = GMMA::K> struct SM90_64x8x16_F32BF16BF16_SS {};
template <GMMA::Major A = GMMA::K, GMMA::Major B = GMMA::K> struct SM90_64x16x16_F32BF16BF16_SS {};
template <GMMA::Major A = GMMA::K, GMMA::Major B = GMMA::K> struct SM90_64x32x16_F32BF16BF16_SS {};
template <GMMA::Major A = GMMA::K, GMMA::Major B = GMMA::K> struct SM90_64x256x16_F32BF16BF16_SS {};
struct SM80_16x8x16_F32BF16BF16F32_TN {};
struct SM80_16x8x16_F16F16F16F16_TN {};
}  // namespace cute

// Top-level GMMA alias used in some headers without `cute::`.
namespace GMMA = cute::GMMA;
template <int N> using Int = cute::Int<N>;

// Variadic make_shape / tile_to_shape stubs used by dispatch even on AMD.
namespace cute {
template <typename... Args> __host__ __device__ inline int make_shape(Args&&...) { return 0; }
template <typename Layout, typename Shape>
__host__ __device__ inline int tile_to_shape(Layout, Shape) { return 0; }
}
using cute::make_shape;
using cute::tile_to_shape;

// CUDA TMA descriptor type — replace with opaque stub on AMD.
using CUtensorMap = struct { uint64_t opaque[16]; };

// AMD-only: structured overlay of CUtensorMap's 128 bytes used to store the
// information the AMD MInst interpreter needs to perform a multi-dimensional
// global→shared copy without relying on a real TMA engine.
//
// magic = 0x54444D41 ('AMDT' little-endian) so the dae2 interpreter can sanity
// check that the descriptor was populated by us (vs. zero-filled or stale).
//
// Layout fits in sizeof(CUtensorMap) = 16 * 8 = 128 bytes (see static_assert
// after the struct). Strides are stored in BYTES so the address math is
//   addr = base_addr + sum_i ( coords[i] * gstride[i] )  for the box origin,
// and the inner box is then walked using bdim[]/elsize.
struct AmdTmaDesc {
  uint32_t magic;          // 0x54444D41 sentinel
  uint8_t  rank;           // 1..5 (matches NVIDIA encodeTiled tensorRank)
  uint8_t  elsize;         // bytes per element (e.g. 2 for bf16/fp16)
  uint16_t reserved;
  uint64_t base_addr;      // global pointer to the tensor base
  uint32_t gdim[5];        // global tensor extents (only first `rank` valid)
  uint64_t gstride[5];     // strides in BYTES; gstride[rank-1] = 0 (unused)
  uint32_t bdim[5];        // box (tile) extents
  uint32_t estride[5];     // element stride within the box (usually 1)
};

static_assert(sizeof(AmdTmaDesc) <= sizeof(CUtensorMap),
              "AmdTmaDesc must fit in 128 bytes (sizeof CUtensorMap)");

constexpr uint32_t AMD_TMA_DESC_MAGIC = 0x54444D41u;  // 'AMDT'

// CUDA driver-API integer typedefs used by torch_runtime.cu's host-side
// TMA-descriptor builder. These are CUDA-specific; on AMD nothing actually
// reads them at runtime (cuTensorMapEncodeTiled is itself stubbed below) but
// the source must compile.
using cuuint32_t = uint32_t;
using cuuint64_t = uint64_t;

// CUtensorMap host-side enum stubs. Real values come from cuda.h on NVIDIA;
// here we just need *some* names with stable integer values so switch/case
// statements compile. The descriptor itself is never used on AMD.
using CUtensorMapDataType    = int;
using CUtensorMapSwizzle     = int;
using CUtensorMapInterleave  = int;
using CUtensorMapL2promotion = int;
using CUtensorMapFloatOOBfill= int;

enum : int {
  CU_TENSOR_MAP_DATA_TYPE_UINT8        = 0,
  CU_TENSOR_MAP_DATA_TYPE_UINT16       = 1,
  CU_TENSOR_MAP_DATA_TYPE_UINT32       = 2,
  CU_TENSOR_MAP_DATA_TYPE_INT32        = 3,
  CU_TENSOR_MAP_DATA_TYPE_UINT64       = 4,
  CU_TENSOR_MAP_DATA_TYPE_INT64        = 5,
  CU_TENSOR_MAP_DATA_TYPE_FLOAT16      = 6,
  CU_TENSOR_MAP_DATA_TYPE_FLOAT32      = 7,
  CU_TENSOR_MAP_DATA_TYPE_BFLOAT16     = 9,

  CU_TENSOR_MAP_SWIZZLE_NONE           = 0,
  CU_TENSOR_MAP_SWIZZLE_32B            = 1,
  CU_TENSOR_MAP_SWIZZLE_64B            = 2,
  CU_TENSOR_MAP_SWIZZLE_128B           = 3,

  CU_TENSOR_MAP_INTERLEAVE_NONE        = 0,
  CU_TENSOR_MAP_INTERLEAVE_16B         = 1,
  CU_TENSOR_MAP_INTERLEAVE_32B         = 2,

  CU_TENSOR_MAP_L2_PROMOTION_NONE      = 0,
  CU_TENSOR_MAP_L2_PROMOTION_L2_64B    = 1,
  CU_TENSOR_MAP_L2_PROMOTION_L2_128B   = 2,
  CU_TENSOR_MAP_L2_PROMOTION_L2_256B   = 3,

  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE    = 0,
};

// Host-side helper: convert CUtensorMapDataType to byte size. Defined after
// the enum constants so the case labels are visible.
__host__ inline uint8_t amd_tma_elsize_from_dtype(int dtype) {
  switch (dtype) {
    case CU_TENSOR_MAP_DATA_TYPE_UINT8:    return 1;
    case CU_TENSOR_MAP_DATA_TYPE_UINT16:   return 2;
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT16:  return 2;
    case CU_TENSOR_MAP_DATA_TYPE_BFLOAT16: return 2;
    case CU_TENSOR_MAP_DATA_TYPE_UINT32:   return 4;
    case CU_TENSOR_MAP_DATA_TYPE_INT32:    return 4;
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT32:  return 4;
    case CU_TENSOR_MAP_DATA_TYPE_UINT64:   return 8;
    case CU_TENSOR_MAP_DATA_TYPE_INT64:    return 8;
    default:                               return 0;  // unknown
  }
}

// Host-only: AMD replacement for the CUDA driver builder. Populates the
// CUtensorMap storage with our AmdTmaDesc layout so the AMD MInst interpreter
// can perform multi-dimensional copies without a real TMA engine.
//
// Swizzle / L2 promotion / OOB fill are NVIDIA-specific perf hints; we record
// nothing and treat the descriptor as "no-swizzle, fill-zero" semantics. This
// is fine for correctness on data that exists; it would skip OOB padding that
// real TMA can do on NVIDIA, but our compute paths on AMD don't rely on that.
__host__ inline hipError_t cuTensorMapEncodeTiled(
    CUtensorMap* tma,
    CUtensorMapDataType dtype,
    cuuint32_t rank,
    void* base,
    const cuuint64_t* global_dims,
    const cuuint64_t* global_strides,
    const cuuint32_t* box_dims,
    const cuuint32_t* element_strides,
    CUtensorMapInterleave,
    CUtensorMapSwizzle,
    CUtensorMapL2promotion,
    CUtensorMapFloatOOBfill) {
  if (tma == nullptr) return hipErrorInvalidValue;
  if (rank == 0 || rank > 5) return hipErrorInvalidValue;
  *tma = CUtensorMap{};
  AmdTmaDesc* d = reinterpret_cast<AmdTmaDesc*>(tma);
  d->magic     = AMD_TMA_DESC_MAGIC;
  d->rank      = static_cast<uint8_t>(rank);
  d->elsize    = amd_tma_elsize_from_dtype(static_cast<int>(dtype));
  d->reserved  = 0;
  d->base_addr = reinterpret_cast<uint64_t>(base);
  for (uint32_t i = 0; i < rank; ++i) {
    d->gdim[i]    = static_cast<uint32_t>(global_dims ? global_dims[i] : 0);
    d->gstride[i] = global_strides ? static_cast<uint64_t>(global_strides[i]) : 0;
    d->bdim[i]    = box_dims ? box_dims[i] : 0;
    d->estride[i] = element_strides ? element_strides[i] : 1;
  }
  for (uint32_t i = rank; i < 5; ++i) {
    d->gdim[i] = 0; d->gstride[i] = 0; d->bdim[i] = 0; d->estride[i] = 0;
  }
  return hipSuccess;
}

// libcu++ tx-counting helpers used in TMA loads. Stubs that trap if reached.
namespace cuda {
template <int Align> struct aligned_size_t {
  size_t value;
  __device__ aligned_size_t(size_t v) : value(v) {}
};
namespace device {
template <typename Bar>
__device__ __forceinline__ void barrier_expect_tx(Bar&, ::cuda::aligned_size_t<16>) { __builtin_trap(); }
template <typename T, typename Bar>
__device__ __forceinline__ void memcpy_async_tx(T*, const T*, ::cuda::aligned_size_t<16>, Bar&) { __builtin_trap(); }
}
}

namespace cuda {

// thread_scope tag — AMD has no equivalent libcu++ enum. Define dummies.
enum thread_scope {
  thread_scope_thread,
  thread_scope_block,
  thread_scope_device,
  thread_scope_system,
};

// ----- cuda::barrier shim -----
//
// Real cuda::barrier<thread_scope_block> is a phase-counted block barrier with
// an expected arrival count set at init(). On AMD we emulate with a single
// shared uint64_t containing { expected:32 | pending:32 } plus a parity bit
// flipped on each phase. arrive() atomically decrements pending; arrive_and_wait
// spins until pending == 0 (then the next phase implicitly resets).
//
// This is *good enough* for queue-style point-to-point handoff between two
// warps in a CTA, which is how VDCores uses it.
template <thread_scope Scope = thread_scope_block>
struct barrier {
  // 32-bit pending counter | 32-bit expected count. Stored in a single
  // 64-bit atomic so we can reset both in one shot at phase rollover.
  unsigned long long state;

  __device__ __forceinline__ void __init(unsigned expected) {
    state = (static_cast<unsigned long long>(expected) << 32) | expected;
  }

  __device__ __forceinline__ uint64_t arrive(unsigned n = 1) {
    auto* p = reinterpret_cast<unsigned long long*>(&state);
    unsigned long long old = atomicAdd(p, -static_cast<long long>(n));
    unsigned pending = static_cast<unsigned>(old & 0xFFFFFFFFULL) - n;
    unsigned expected = static_cast<unsigned>(old >> 32);
    if (pending == 0) {
      // Reset pending = expected for next phase.
      atomicAdd(p, static_cast<long long>(expected));
    }
    return static_cast<uint64_t>(old);
  }

  __device__ __forceinline__ void wait(uint64_t /*token*/) {
    // atomicAdd(p, 0) is the bulletproof spin-read on AMD HIP: a plain
    // load (or even a volatile load) can still be hoisted past s_sleep
    // by the backend, but an atomic op is a hard memory barrier that
    // forces a fresh fetch from LDS on every iteration.
    auto* p = reinterpret_cast<unsigned long long*>(&state);
    while (true) {
      unsigned long long s = atomicAdd(p, 0ULL);
      unsigned pending  = static_cast<unsigned>(s & 0xFFFFFFFFULL);
      unsigned expected = static_cast<unsigned>(s >> 32);
      if (pending == expected) break;  // released to next phase
      __builtin_amdgcn_s_sleep(1);
    }
  }

  __device__ __forceinline__ void arrive_and_wait() {
    uint64_t tok = arrive(1);
    wait(tok);
  }
};

namespace device {
template <thread_scope Scope>
__device__ __forceinline__ uint64_t* barrier_native_handle(barrier<Scope>& b) {
  return reinterpret_cast<uint64_t*>(&b.state);
}
}  // namespace device

// init(&barrier, expected_count) — analog of cuda::barrier_init / cuda::std::init
template <thread_scope Scope>
__device__ __forceinline__ void barrier_init(barrier<Scope>& b, unsigned expected) {
  b.__init(expected);
}
template <thread_scope Scope>
__device__ __forceinline__ void init(barrier<Scope>* b, unsigned expected) {
  b->__init(expected);
}

// ----- cuda::ptx shim -----
namespace ptx {

// integer constant template (used by cp_async_bulk_wait_group<N>())
template <int N>
struct n32_t {};

// Special-register helpers.
__device__ __forceinline__ unsigned get_sreg_laneid() {
  return __builtin_amdgcn_mbcnt_hi(-1u, __builtin_amdgcn_mbcnt_lo(-1u, 0));
}
__device__ __forceinline__ unsigned get_sreg_clusterid_x() {
  return blockIdx.x;
}
__device__ __forceinline__ unsigned long long get_sreg_globaltimer() {
  // Wave-comparable wall-clock counter on CDNA. __builtin_readcyclecounter
  // (s_memtime) is per-shader-engine and yields garbage when start and end
  // are written from different waves — use s_memrealtime (the steady counter
  // backing wall_clock64) so end-start is a valid positive delta.
  // Unit is gfx942 wall-clock ticks (typically 100 MHz → 10 ns/tick); the
  // host queries hipDeviceAttributeWallClockRate to convert to ns.
  return __builtin_amdgcn_s_memrealtime();
}

// Memory-space tags (NVIDIA TMA API). On AMD we just need names.
struct space_global_t {};
struct space_shared_t {};
inline constexpr space_global_t  space_global{};
inline constexpr space_shared_t  space_shared{};

// ----- TMA / cp_async_bulk: STUBBED -----
//
// On AMD there is no direct TMA equivalent. Calling these traps the wave so
// any kernel that actually exercises a TMA path fails loudly rather than
// silently producing garbage. The build still succeeds.
template <typename... Args>
__device__ __forceinline__ void cp_async_bulk(space_global_t, space_shared_t,
                                               Args&&...) {
  __builtin_trap();  // VDCores TMA load/store: not implemented on AMD
}
template <typename... Args>
__device__ __forceinline__ void cp_async_bulk(space_shared_t, space_global_t,
                                               Args&&...) {
  __builtin_trap();
}
__device__ __forceinline__ void cp_async_bulk_commit_group() {
  // No-op: there is no group to commit. A trap on cp_async_bulk above will
  // already have fired before we reach here.
}
template <int N>
__device__ __forceinline__ void cp_async_bulk_wait_group(n32_t<N>) {}
template <int N>
__device__ __forceinline__ void cp_async_bulk_wait_group_read(n32_t<N>) {}

// fence_proxy_async — degrades to a block-scope threadfence on AMD.
__device__ __forceinline__ void fence_proxy_async() { __threadfence_block(); }
__device__ __forceinline__ void fence_proxy_async(space_shared_t) { __threadfence_block(); }

}  // namespace ptx
}  // namespace cuda

// __cvta_generic_to_shared is a CUDA builtin used to convert a generic shared
// pointer to a 32-bit shared address. AMD has no notion of this — the address
// space is unified for our purposes — so just truncate the pointer.
#ifndef __cvta_generic_to_shared
#define __cvta_generic_to_shared(p) (static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p)))
#endif

#endif  // __HIP_PLATFORM_AMD__
