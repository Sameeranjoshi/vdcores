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
  const int wave = tid / 64;   // 0,1 = compute; 2 = LD; 3 = ST
  const int lane = tid % 64;

  // ---- Shared state (declared up-front so all 256 threads init together) ----
  __shared__ DaeAmdSignal load_done;
  __shared__ DaeAmdSignal store_done;
  __shared__ DaeAmdSignal compute_done;
  // Two LDS staging slots A,B (16 KB each = 32 KB total). Single-input tests
  // (smoke / tmacopy / tma1d) use slot A only; SILU and other dual-input ops
  // use both.
  __shared__ alignas(16) uint8_t lds_slot_a[daeAmdStagingBytes];
  __shared__ alignas(16) uint8_t lds_slot_b[daeAmdStagingBytes];
  // Compute-mode tag set by the kernel-start scan. The LD/ST waves use this
  // to decide whether to read from slot B (multi-input ops) and whether to
  // wait on compute_done (compute-producing ops) instead of load_done.
  enum ComputeMode : uint8_t { CMODE_NONE = 0, CMODE_SILU = 1, CMODE_MFMA = 2 };
  __shared__ uint8_t  compute_mode;
  __shared__ uint16_t silu_num_token;
  __shared__ unsigned compute_progress;  // counter used by compute wave to fence its writes

  // AMD MFMA bf16 16x16 matmul demo: hijack OP_GEMM_M64N64 (an opcode that
  // is in the selected compute set so the Python launcher's validation
  // passes, but unused by any AMD-target python test today). Re-purposing
  // it lets us exercise the full Python launcher path without adding a new
  // op to the generated compute_opcode_order.inc.
  constexpr uint16_t OP_AMD_DEBUG_MATMUL_BF16_16x16 = OP_GEMM_M64N64;

  // Init: thread 0 zeroes signals + scans cinsts for compute-op detection.
  if (tid == 0) {
    load_done.counter = 0;
    store_done.counter = 0;
    compute_done.counter = 0;
    compute_mode = CMODE_NONE;
    silu_num_token = 0;
    compute_progress = 0;

    const CInst* cinsts = compute_instructions + sm_id * numInsts;
    for (uint32_t pc = 0; pc < numInsts; ++pc) {
      uint16_t opc = cinsts[pc].opcode;
      if (opc == OP_TERMINATEC) break;
      if (opc == OP_SILU_MUL_SHARED_BF16_K_4096_INTER) {
        compute_mode    = CMODE_SILU;
        silu_num_token  = cinsts[pc].args[0];
        break;
      }
      if (opc == OP_AMD_DEBUG_MATMUL_BF16_16x16) {
        compute_mode = CMODE_MFMA;
        break;
      }
    }
  }
  __syncthreads();   // all 256 threads see the init state
  const bool is_silu_mode = (compute_mode == CMODE_SILU);
  const bool is_mfma_mode = (compute_mode == CMODE_MFMA);
  const bool produces_compute = (compute_mode != CMODE_NONE);
  const bool needs_two_inputs = (compute_mode != CMODE_NONE);

  // ---- Compute group (threads 0..127, waves 0+1) ----
  if (tid < numComputeWarps * 32) {
    if (is_silu_mode) {
      // Wait for both inputs (gate→A, up→B) to finish loading.
      while (atomicAdd(&load_done.counter, 0u) < 2u) {
        __builtin_amdgcn_s_sleep(1);
      }

      // SILU + Mul: out[i] = silu(gate[i]) * up[i], in-place into slot A.
      // Element layout matches silu_mul.py: K=4096 bf16 elements per token,
      // packed as bf16x2 (so K/2 packed pairs per token), N tokens.
      const int K = 4096;                     // INTERM_DIM in silu_mul.py
      const int N = silu_num_token;
      const int total = (K / 2) * N;          // bf162 elements
      const int n_compute_threads = numComputeWarps * 32;

      __hip_bfloat162* sGate = reinterpret_cast<__hip_bfloat162*>(lds_slot_a);
      __hip_bfloat162* sUp   = reinterpret_cast<__hip_bfloat162*>(lds_slot_b);
      __hip_bfloat162* sOut  = reinterpret_cast<__hip_bfloat162*>(lds_slot_a);

      for (int i = tid; i < total; i += n_compute_threads) {
        __hip_bfloat162 g = sGate[i];
        __hip_bfloat162 u = sUp[i];
        float gx = float(g.x), gy = float(g.y);
        float ux = float(u.x), uy = float(u.y);
        float ox = (gx / (1.0f + expf(-gx))) * ux;
        float oy = (gy / (1.0f + expf(-gy))) * uy;
        __hip_bfloat162 r;
        r.x = __hip_bfloat16(ox);
        r.y = __hip_bfloat16(oy);
        sOut[i] = r;
      }

      // Inter-wave fence inside the compute group: every compute thread bumps
      // the progress counter, then spins until all 128 have arrived. This
      // ensures slot A is fully written before we signal compute_done.
      __threadfence_block();
      atomicAdd(&compute_progress, 1u);
      while (atomicAdd(&compute_progress, 0u) < (unsigned)n_compute_threads) {
        __builtin_amdgcn_s_sleep(1);
      }

      if (tid == 0) {
        __threadfence_block();
        dae_amd_arrive(compute_done);
      }
    } else if (is_mfma_mode) {
      // bf16 16x16x16 matmul on a single wave64 wavefront via MFMA.
      // Inputs: slot A = A[16,16] bf16 row-major, slot B = B[16,16] bf16 row-major.
      // Output: D = A @ B as bf16 row-major, written back into slot A in-place.
      // Lane mapping verified by app/hip/mfma_matmul.cpp (job 303883).

      // Wait for both inputs.
      while (atomicAdd(&load_done.counter, 0u) < 2u) {
        __builtin_amdgcn_s_sleep(1);
      }

      // Only wave 0 (lanes 0..63) participates in the MFMA. Wave 1 sits idle.
      typedef int16_t bf16x4 __attribute__((ext_vector_type(4)));
      typedef float   f32x4  __attribute__((ext_vector_type(4)));

      if (tid < 64) {
        const int lane = tid;
        const int kblk = lane / 16;   // 0..3
        const int row  = lane % 16;   // 0..15 (M index for A operand)
        const int col  = lane % 16;   // 0..15 (N index for B operand)

        // Read row-major sA / sB from LDS (each is 16x16 bf16 = 512 bytes).
        const __hip_bfloat16* sA = reinterpret_cast<const __hip_bfloat16*>(lds_slot_a);
        const __hip_bfloat16* sB = reinterpret_cast<const __hip_bfloat16*>(lds_slot_b);

        bf16x4 a, b;
        // Memcpy the 16-bit bit-pattern through to keep the conversion safe.
        for (int i = 0; i < 4; ++i) {
          __hip_bfloat16 av = sA[row * 16 + kblk * 4 + i];
          __hip_bfloat16 bv = sB[(kblk * 4 + i) * 16 + col];
          int16_t ai, bi;
          __builtin_memcpy(&ai, &av, sizeof(ai));
          __builtin_memcpy(&bi, &bv, sizeof(bi));
          a[i] = ai;
          b[i] = bi;
        }

        f32x4 c = {0.0f, 0.0f, 0.0f, 0.0f};
        f32x4 d = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);

        // Cast each lane's 4 fp32 outputs back to bf16 and write into slot A
        // in row-major order. lane k holds D[(k/16)*4 + 0..3, k%16].
        const int mblk = lane / 16;
        const int n    = lane % 16;
        __hip_bfloat16* sOut = reinterpret_cast<__hip_bfloat16*>(lds_slot_a);
        sOut[(mblk * 4 + 0) * 16 + n] = __hip_bfloat16(d[0]);
        sOut[(mblk * 4 + 1) * 16 + n] = __hip_bfloat16(d[1]);
        sOut[(mblk * 4 + 2) * 16 + n] = __hip_bfloat16(d[2]);
        sOut[(mblk * 4 + 3) * 16 + n] = __hip_bfloat16(d[3]);
      }

      // Inter-wave fence (same as SILU path) so all 128 compute threads
      // arrive before signaling compute_done. Wave 1 just bumps the counter.
      const int n_compute_threads = numComputeWarps * 32;
      __threadfence_block();
      atomicAdd(&compute_progress, 1u);
      while (atomicAdd(&compute_progress, 0u) < (unsigned)n_compute_threads) {
        __builtin_amdgcn_s_sleep(1);
      }

      if (tid == 0) {
        __threadfence_block();
        dae_amd_arrive(compute_done);
      }
    } else {
      // No-op: walk cinsts looking for OP_TERMINATEC. OP_DUMMY/OP_COPY are
      // ignored — the LD/ST waves below move all data.
      const CInst* cinsts = compute_instructions + sm_id * numInsts;
      for (uint32_t pc = 0; pc < numInsts; ++pc) {
        uint16_t opc = cinsts[pc].opcode;
        if (opc == OP_TERMINATEC) break;
      }
    }
    return;
  }

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
        // Pick destination slot. Compute ops with two inputs dual-load into
        // A then B; single-input tests use slot A only.
        uint8_t* slot_ptr = (needs_two_inputs && load_seq == 2) ? lds_slot_b : lds_slot_a;

        // Back-pressure (single-slot reuse): wait for the prior ST to drain
        // the slot before reusing it. Compute-mode kernels run one round per
        // kernel launch and don't reuse slots — skip the wait there.
        if (!produces_compute && store_seq > 0) {
          dae_amd_wait_at_least(store_done, load_seq - 1);
        }
        const uint64_t addr = inst.address + addr_offset;
        const uint32_t n    = inst.size;
        dae_amd_copy_g2l(slot_ptr, reinterpret_cast<const void*>(addr), n);
        dae_amd_arrive(load_done);
      }
    } else if (is_alloc_store) {
      store_seq++;
      if (wave == 3 && lane == 0) {
        // Compute-producing modes: wait for the compute to finish writing
        // slot A. Pure memory-pipeline tests: wait for the matching load.
        if (produces_compute) {
          dae_amd_wait_at_least(compute_done, store_seq);
        } else {
          dae_amd_wait_at_least(load_done, store_seq);
        }
        const uint64_t addr = inst.address + addr_offset;
        const uint32_t n    = inst.size;
        dae_amd_copy_l2g(reinterpret_cast<void*>(addr), lds_slot_a, n);
        // device-scope fence so the host (or downstream CTAs) observes the write
        __threadfence();
        dae_amd_arrive(store_done);
      }
    } else if (opc == op(OP_ALLOC_WB_REG_STORE) ||
               opc == op(OP_ALLOC_REG_LOAD)     ||
               opc == op(OP_ALLOC_WB_RAW_ADDRESS)) {
      // Slot/register-file abstractions from the NVIDIA path. The AMD
      // interpreter uses a single LDS staging buffer (no slot pool), so
      // these don't move data — treat them as no-ops so tests like
      // register.py / rmsnorm.py compile and run without crashing.
      // No data movement, no barrier — fall through to pc++.
    } else {
      // Unsupported opcode for this minimal interpreter. Trap so the failure
      // is loud rather than silent. Extend this switch when you need more
      // opcodes (multi-D TMA, BARRIER, etc.).
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
