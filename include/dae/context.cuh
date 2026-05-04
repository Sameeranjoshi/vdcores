#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>
#include "dae/hip_compat.cuh"

// features
constexpr bool dae2EnableLooping = true;
constexpr bool dae2EnableGroup = true;
constexpr bool dae2BlockingStore = false;

#ifdef __HIP_PLATFORM_AMD__
// AMD CDNA has 64 KB LDS per workgroup vs Hopper's 228 KB. The slot pool
// alone (24 * 8 KB = 192 KB on the NVIDIA path) doesn't fit. The AMD
// interpreter uses its own static LDS staging slots (slot A + slot B) in
// addition to the dynamic pool reserved by the Python launcher. Total LDS
// per workgroup must stay under 64 KB:
//   dynamic = numSlots * slotSizeKb * 1024  +  4 KB slack (set in launcher.py)
//   static  = daeAmdStagingBytesA + daeAmdStagingBytesB
// With numSlots=1, slotSizeKb=8, A=32K, B=16K:
//   8 KB dyn + 4 KB slack + 32 KB + 16 KB static = 60 KB  (fits, 4 KB margin).
constexpr bool dae2LoadInstructions = false;
static constexpr int slotSizeKb = 8;
static constexpr int numSlots = 1;
static constexpr int numInsts = 4096;
// Asymmetric staging slots:
//   Slot A is sized to hold 64 rows × 256 cols of bf16 (= 32 KB) so the
//   AMD interpreter can run M=64, K=256 GEMV-class kernels (matching the
//   upstream Gemv_M64N8 atom shape). Smaller workloads (smoke / tmacopy /
//   silu) only use the first 16 KB.
//   Slot B is 16 KB — covers silu_mul's 16 KB "up" tensor and any K=256
//   gemv B (256 × 16 × 2 = 8 KB).
static constexpr int daeAmdStagingBytesA = 32 * 1024;
static constexpr int daeAmdStagingBytesB = 16 * 1024;
#else
constexpr bool dae2LoadInstructions = true;
static constexpr int slotSizeKb = 8;
static constexpr int numSlots = 24;
static constexpr int numInsts = dae2LoadInstructions ? 512 : 4096;
#endif
static constexpr int numTmas = 1024;
static constexpr int numBars = 1024;

static constexpr int numSpecialSlots = 9;

static_assert(numSlots + numSpecialSlots <= ((2<<6) - 1), "Total number of slots must be less than or equal to 32");

static constexpr int numComputeWarps = 4;
static constexpr int numMemoryWarps = 4;

static constexpr int numThreadsPerWarp = 32;
static constexpr int numThreads = numThreadsPerWarp * (numComputeWarps + numMemoryWarps);
// one warpgroup + 1 memory warp
static constexpr int numProfileEvents = 128;
static constexpr int numComputeLoopCounters = 4;

// barrier configurations
static constexpr int numThreadsM2CBarrier = numComputeWarps * numThreadsPerWarp + 1;
static constexpr int numThreadsC2MBarrier = numComputeWarps * numThreadsPerWarp + 1;
static constexpr int numThreadsLDBarrier = 2;

// Polling backoff for the memory core hot loops.
static constexpr int allocRetrySleepCycles = 16;
static constexpr int barrierPollSleepCycles = 16;

// Allocwarp instruction prefetch policy.
static constexpr int allocwarpInstructionPrefetchDistance = 2;
static constexpr int allocwarpInstructionSeedCount = 2;
static constexpr int allocwarpInstructionTargetSpan = 2;

constexpr int flagBits = 6;
constexpr int slotBits = 6;
static_assert(numSlots <= (1 << slotBits), "numSlots exceeds slotBits capacity");

// definition of instruction formats
struct alignas(8) CInst {
  uint16_t opcode;
  uint16_t args[3];
};


// we reserve the lower 6 bit of opcode as decode bits
enum InstOpDecode : uint16_t {
  MEM_OP_FLAGS_NONE = 0x0,
  MEM_OP_FLAGS_ALLOCATE = 0x1,
  MEM_OP_FLAGS_WRITEBACK = 0x2,
  MEM_OP_FLAGS_GROUP = 0x4,
  MEM_OP_FLAGS_JUMP = 0x8,
  MEM_OP_FLAGS_BARRIER = 0x10,
  MEM_OP_FLAGS_PORT = 0x20,
};

enum InstOpDecodeMask : uint16_t {
  MEM_OP_MASK_FLAGS = (1U << flagBits) - 1,
  MEM_OP_MASK_PENDING = 0x0003,
};

static __device__ __host__ __forceinline__ constexpr uint16_t rmask(const uint16_t mask) {
  return (uint16_t)(~mask);
}

#define MK_MOP(opcode, flags) \
    ((uint16_t)(((opcode) << flagBits) | ((flags) & ((1U << flagBits) - 1))))
    
enum InstOpcode : uint16_t {
  #define DAE_OP(name, value) name = value,
    #include "dae/opcode.cuh.inc"
  #undef DAE_OP
};

// TODO(zhiyuang): load128
struct alignas(16) MInst {
  uint16_t opcode; // 12 bits opcode + 4 bits flags
  uint16_t size;
  union {
    struct {
      uint16_t num_slots;
      uint16_t arg;
    };
    uint32_t shifter; // for shifting the address or arg field
  };

  union {
    uint64_t address;     // For other purpose
    uint16_t coords[4];   // For up to 4D TMA coordinates
  };

  __device__ __forceinline__ uint16_t flag(const uint16_t f) const {
    return opcode & f;
  }
  __device__ __forceinline__ uint16_t nslot() const {
    constexpr uint16_t slotMask = (1U << slotBits) - 1;
    return num_slots & slotMask;
  }
  __device__ __forceinline__ uint16_t bar() const {
    return num_slots >> slotBits;
  }
};

// helpers for building opcode
static __device__ __host__ constexpr uint16_t op(const uint16_t opcode) {
  return opcode >> flagBits;
}

static __device__ __host__ constexpr uint16_t jump(const uint16_t opcode) {
  return opcode | MEM_OP_FLAGS_JUMP;
}
