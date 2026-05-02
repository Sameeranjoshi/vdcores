#include "hip/hip_runtime.h"
#pragma once

#include "virtualcore.cuh"

#include "allocator.cuh"
#include "queue.cuh"
#include "compute_dispatch.cuh"

#include <hip/hip_runtime.h>
#include "dae/hip_compat.cuh"
#include <bit>

// pipeline stages
#include "pipeline/allocwarp.cuh"
#include "pipeline/ldwarp.cuh"
#include "pipeline/stwarp.cuh"

static __device__ __forceinline__ void * align_to(void *ptr, size_t align) {
  uintptr_t addr = (uintptr_t)ptr;
  uintptr_t aligned = (addr + align - 1) & ~(align - 1);
  return (void*)aligned;
}

// TODO(zhiyuang): decide this maxnreg size.
// Also with setnreg for computation and memory separately?
// ============================================================================
// AMD path: minimal MInst interpreter using L3-style wave64-safe topology.
//
// Replaces the upstream megakernel body (which deadlocks on AMD because
// alloc and ST share a wave64 wavefront — see
// docs/superpowers/plan-b-spike-findings-2026-05-01.md). For dae_copy_smoke.py
// and any test using only OP_REPEAT / OP_ALLOC_TMA_LOAD_1D /
// OP_ALLOC_WB_TMA_STORE_1D / OP_TERMINATE on the mem side, this interpreter
// produces the same observable output as the upstream megakernel would on
// NVIDIA. Other opcodes intentionally trap so unsupported tests fail loudly.
//
// Block layout (uses upstream numThreads = 256 unchanged, so launch_dae and
// the Python launcher stay byte-identical):
//   threads 0  .. 127  → compute warps; walk cinsts to OP_TERMINATEC and exit.
//                        OP_DUMMY / OP_COPY become no-ops since the LD/ST
//                        waves drive data movement directly.
//   threads 128.. 191  → LD wave (wave64). Lane 0 issues global→LDS copies.
//   threads 192.. 255  → ST wave (wave64). Lane 0 issues LDS→global drains.
// LD and ST live in distinct wave64 wavefronts, so EXEC-mask serialization
// can't deadlock them against each other.
// ============================================================================
#ifdef __HIP_PLATFORM_AMD__
struct DaeAmdSignal {
  unsigned counter;
};

__device__ __forceinline__
static void dae_amd_arrive(DaeAmdSignal& s) {
  __threadfence_block();
  atomicAdd(&s.counter, 1u);
}

__device__ __forceinline__
static void dae_amd_wait_at_least(DaeAmdSignal& s, unsigned target) {
  while (atomicAdd(&s.counter, 0u) < target) {
    __builtin_amdgcn_s_sleep(1);
  }
}

// Wide LDS↔global memcpy gated by alignment. Single-thread driven (lane 0 of
// the calling wave). Falls back to byte copy on unaligned addresses or sizes.
__device__ __forceinline__
static void dae_amd_copy_g2l(void* lds_dst, const void* g_src, uint32_t n) {
  uintptr_t s = reinterpret_cast<uintptr_t>(g_src);
  uintptr_t d = reinterpret_cast<uintptr_t>(lds_dst);
  if (((s | d) & 0xF) == 0 && (n & 0xF) == 0) {
    const uint4* sp = reinterpret_cast<const uint4*>(g_src);
    uint4* dp = reinterpret_cast<uint4*>(lds_dst);
    uint32_t n16 = n >> 4;
    for (uint32_t i = 0; i < n16; ++i) dp[i] = sp[i];
  } else {
    const uint8_t* sp = reinterpret_cast<const uint8_t*>(g_src);
    uint8_t* dp = reinterpret_cast<uint8_t*>(lds_dst);
    for (uint32_t i = 0; i < n; ++i) dp[i] = sp[i];
  }
}

__device__ __forceinline__
static void dae_amd_copy_l2g(void* g_dst, const void* lds_src, uint32_t n) {
  uintptr_t s = reinterpret_cast<uintptr_t>(lds_src);
  uintptr_t d = reinterpret_cast<uintptr_t>(g_dst);
  if (((s | d) & 0xF) == 0 && (n & 0xF) == 0) {
    const uint4* sp = reinterpret_cast<const uint4*>(lds_src);
    uint4* dp = reinterpret_cast<uint4*>(g_dst);
    uint32_t n16 = n >> 4;
    for (uint32_t i = 0; i < n16; ++i) dp[i] = sp[i];
  } else {
    const uint8_t* sp = reinterpret_cast<const uint8_t*>(lds_src);
    uint8_t* dp = reinterpret_cast<uint8_t*>(g_dst);
    for (uint32_t i = 0; i < n; ++i) dp[i] = sp[i];
  }
}
#endif  // __HIP_PLATFORM_AMD__

// TODO(zhiyuang): decide this maxnreg size.
// Also with setnreg for computation and memory separately?
static __global__
void dae2(
  const CInst* __restrict__ compute_instructions,
  const MInst* __restrict__ memory_instructions,
  const CUtensorMap* __restrict__ tma_descs,
  int * __restrict__ bars,
  uint64_t *  __restrict__ g_events
) {
#ifdef __HIP_PLATFORM_AMD__
  // ---- AMD interpreter ----
  const int sm_id = blockIdx.x;
  const int tid = threadIdx.x;

  // Compute group: threads 0..numComputeWarps*32-1. Walk cinsts to TERMINATEC.
  if (tid < numComputeWarps * 32) {
    const CInst* cinsts = compute_instructions + sm_id * numInsts;
    for (uint32_t pc = 0; pc < numInsts; ++pc) {
      uint16_t opc = cinsts[pc].opcode;
      if (opc == OP_TERMINATEC) break;
      // OP_DUMMY / OP_COPY / others: no-op. The LD/ST waves below move all data.
    }
    return;
  }

  // Memory group: 64-thread "LD wave" at threads 128-191 and "ST wave" at 192-255.
  const int wave = tid / 64;     // 2 = LD, 3 = ST
  const int lane = tid % 64;

  __shared__ DaeAmdSignal load_done;
  __shared__ DaeAmdSignal store_done;
  // Single-buffer LDS staging slot for the AMD interpreter. Sized via
  // daeAmdStagingBytes (16 KB) so it fits the largest workload chunk we
  // care about (tmacopy.py uses 16 KB loads). Independent of the Python
  // slot pool — see context.cuh for the full LDS budget breakdown.
  __shared__ alignas(16) uint8_t lds_slot[daeAmdStagingBytes];

  if (tid == 128) {
    load_done.counter = 0;
    store_done.counter = 0;
  }
  __syncthreads();

  const MInst* minsts = memory_instructions + sm_id * numInsts;

  uint32_t pc = 0;
  uint32_t loop_counter = 0;
  uint32_t loop_start_pc = 0;
  uint64_t repeat_offset = 0;     // gpr[0] in the upstream model
  uint64_t addr_offset   = 0;     // gpr[1] accumulator
  uint32_t load_seq  = 0;         // monotonic counter on load_done
  uint32_t store_seq = 0;         // monotonic counter on store_done

  // Safety bound — interpreter exits via OP_TERMINATE; this just prevents
  // runaway hangs on a malformed program.
  for (uint32_t safety = 0; safety < numInsts * 8u; ++safety) {
    if (pc >= numInsts) break;
    MInst inst = minsts[pc];
    uint16_t opcode = inst.opcode;
    uint16_t opc = op(opcode);

    if (opc == op(OP_TERMINATE)) break;

    if (opc == op(OP_REPEAT)) {
      loop_counter  = inst.size;
      loop_start_pc = pc + 1;
      repeat_offset = inst.address;
      addr_offset   = 0;
      pc++;
      continue;
    }

    bool is_alloc_load  = (opc == op(OP_ALLOC_TMA_LOAD_1D));
    bool is_alloc_store = (opc == op(OP_ALLOC_WB_TMA_STORE_1D));

    if (is_alloc_load) {
      load_seq++;
      if (wave == 2 && lane == 0) {
        // Wait for prev ST to drain the slot before reusing it.
        dae_amd_wait_at_least(store_done, load_seq - 1);
        const uint64_t addr = inst.address + addr_offset;
        const uint32_t n    = inst.size;
        dae_amd_copy_g2l(lds_slot, reinterpret_cast<const void*>(addr), n);
        dae_amd_arrive(load_done);
      }
    } else if (is_alloc_store) {
      store_seq++;
      if (wave == 3 && lane == 0) {
        dae_amd_wait_at_least(load_done, store_seq);
        const uint64_t addr = inst.address + addr_offset;
        const uint32_t n    = inst.size;
        dae_amd_copy_l2g(reinterpret_cast<void*>(addr), lds_slot, n);
        // device-scope fence so the host (or downstream CTAs) observes the write
        __threadfence();
        dae_amd_arrive(store_done);
      }
    } else {
      // Unsupported opcode for this minimal interpreter. Trap so the failure
      // is loud rather than silent. Extend this switch when you need more
      // opcodes (multi-D TMA, REG_LOAD/STORE, BARRIER, etc.).
      __builtin_trap();
    }

    pc++;

    // Loop jump: the LAST instruction of a repeat body has the JUMP flag.
    // Decrement loop_counter; if more iterations remain, jump back and
    // bump the address accumulator by repeat_offset.
    if ((opcode & MEM_OP_FLAGS_JUMP) && loop_counter > 0) {
      --loop_counter;
      if (loop_counter > 0) {
        pc = loop_start_pc;
        addr_offset += repeat_offset;
      }
    }
  }
  return;
#else
  // ---- NVIDIA path: original megakernel ----
  int sm_id = blockIdx.x;
  int thread_id = threadIdx.x;
  int warp_id = (thread_id % 128) / 32;
  int lane_id = thread_id % 32;


  __kprint("[DAE2 SM %d] Kernel launched with %d threads (%d warps)\n", sm_id, blockDim.x, blockDim.x / 32);


  const CInst* __restrict__ cinsts;
  const MInst* __restrict__ minsts;

  // local datastructures
  if constexpr (dae2LoadInstructions) {
    __shared__ CInst smem_cinsts[numInsts];
    __shared__ MInst smem_minsts[numInsts];

    for (int i = thread_id; i < numInsts; i += blockDim.x) {
      smem_cinsts[i] = compute_instructions[sm_id * numInsts + i];
      smem_minsts[i] = memory_instructions[sm_id * numInsts + i];
    }

    cinsts = smem_cinsts;
    minsts = smem_minsts;
  } else {
    cinsts = compute_instructions + sm_id * numInsts;
    minsts = memory_instructions + sm_id * numInsts;
  }

  // intermidate insts
  constexpr int numQueueElements = 32;
  __shared__ MInst st_insts[numSlots + numSpecialSlots]; // we can have some special slots that don't go through the allocator, for special purposes like reduction output, argmax output, etc. these are indexed from numSlots and above.

  // allocator
  // TODO(zhiyuang): align this to lane 31 to avoid bank conflict?
  __shared__ int slot_avail;
  if (thread_id == 0)
    slot_avail = (1U << numSlots) - 1; // all slots are available at the beginning. each bit represents a slot. 1 means available, 0 means occupied.

  // Init the queues
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ cuda::barrier<cuda::thread_scope_block> barriers[4][numQueueElements];
  assert(numQueueElements <= blockDim.x && "Too many slots for barriers");
  if (threadIdx.x < numQueueElements) {
    init(&barriers[0][threadIdx.x], numThreadsM2CBarrier);
    init(&barriers[1][threadIdx.x], numThreadsC2MBarrier);
    init(&barriers[2][threadIdx.x], numThreadsLDBarrier);
    init(&barriers[3][threadIdx.x], numThreadsLDBarrier);
  }

  __shared__ int m2c_data[numQueueElements];
  __shared__ int c2m_data[numQueueElements];
  __shared__ int m2ld_data[2][numQueueElements];

  SizeBoundedBarrierQueue<int, numQueueElements> m2c {
    .barriers = barriers[0], .data = m2c_data, .ptr = 0
  };
  SizeBoundedBarrierAllocQueue<numQueueElements> c2m {
    barriers[1], c2m_data, 0, &slot_avail
  };
  SizeBoundedBarrierQueue<int, numQueueElements> m2ld[2] = {
    { .barriers = barriers[2], .data = m2ld_data[0], .ptr = 0 },
    { .barriers = barriers[3], .data = m2ld_data[1], .ptr = 0 }
  };

  // init the slots
  extern __shared__ uint8_t shared_mem[];
  void * smem_base = align_to((void*)shared_mem, 1024); // align to 1KB

  // alloc a small scratch space for temporary data
  // argmax uses this
  __shared__ uint64_t scratch_space[32]; // 8-bytes aligned

  if (threadIdx.x == 0) {
    int event_base = sm_id * numProfileEvents;
    g_events[event_base + 0] = cuda::ptx::get_sreg_globaltimer();
  }

  __syncthreads();

  // start memory and computation execution
  if (threadIdx.x < numComputeWarps * 32) {
    CInst inst;
    uint32_t pc = 0;
    uint32_t count[numComputeLoopCounters] = {};
    bool finish = false;

    while (!finish) {
      inst = cinsts[(pc++) % numInsts];
    
      __cprint("Executing instruction at PC %d: opcode=%04x", pc - 1, inst.opcode);
      dispatch_compute_instruction(
        sm_id,
        thread_id,
        pc,
        count,
        finish,
        inst,
        smem_base,
        scratch_space,
        st_insts,
        m2c,
        c2m,
        g_events
      );
      // if (blockIdx.x == 0 && threadIdx.x == 0) {
      //   printf("[COMP] after execution: pc=%d, opcode=%04x\n", pc-1, inst.opcode);
      // }
    }
    __cprint("Finished execution pc=%d", pc-1);
  } else { // memory warp group
    // TODO(zhiyuang): reduce the register usage in memory warps
    // cuda::ptx::set_max_nreg();

#if defined(__HIP_PLATFORM_AMD__) && defined(DAE_DEBUG_PRINT)
    if (blockIdx.x == DAE_DEBUG_PRINT) {
      unsigned lid = cuda::ptx::get_sreg_laneid();
      if (lid < 2)
        printf("[%d][DISP] mem-group: tx=%d warp_id=%d lane_id=%d lid=%u\n",
               (int)blockIdx.x, (int)threadIdx.x, warp_id, lane_id, lid);
    }
#endif

    // TODO(zhiyuang): change this to threadIdx.x predicates. will be faster than lane_id based?
#if defined(__HIP_PLATFORM_AMD__) && defined(DAE_DEBUG_PRINT)
    if (threadIdx.x == 128) {
      printf("[%d][PRE-DISPATCH] thread 128 about to enter warp dispatch warp_id=%d\n",
             (int)blockIdx.x, warp_id);
    }
#endif
    if (warp_id == 0) {
#if defined(__HIP_PLATFORM_AMD__) && defined(DAE_DEBUG_PRINT)
      if (threadIdx.x == 128) {
        printf("[%d][PRE-ALLOC-CALL] thread 128 about to call allocwarp_execute\n",
               (int)blockIdx.x);
      }
#endif
      allocwarp_execute(
        lane_id,
        m2c, m2ld, minsts, &slot_avail,
        st_insts, smem_base, tma_descs, bars
      );
#if defined(__HIP_PLATFORM_AMD__) && defined(DAE_DEBUG_PRINT)
      if (threadIdx.x == 128) {
        printf("[%d][POST-ALLOC-CALL] thread 128 returned from allocwarp_execute\n",
               (int)blockIdx.x);
      }
#endif
    } else if (warp_id == 1) {
      if (lane_id == 0) {
        stwarp_execute_singlethread(
          c2m, st_insts,
          smem_base, tma_descs, bars
        );
      }
    } else if (warp_id >= 2) { // LD Warps 0-1
      if (lane_id == 0) {
        int port_id = warp_id - 2;
        ldwarp_execute_singlethread(
          m2ld[port_id], m2c,
          st_insts,
          smem_base, tma_descs, bars
        );
      }
    } // End of warps
  } // End of memory warp group

  // end of megakernel
#endif  // __HIP_PLATFORM_AMD__ (else branch above)
}
