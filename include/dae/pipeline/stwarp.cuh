#include "hip/hip_runtime.h"
#pragma once

#include "virtualcore.cuh"

#ifdef __HIP_PLATFORM_AMD__
// AMD: real ST pipeline for the 1-D LDS -> global path used by tmacopy.py.
// Hopper's cp_async_bulk + commit_group / wait_group is replaced with a
// synchronous wide-store loop. We threadfence before the store so prior LDS
// writes from compute warps are visible, and threadfence_system after the
// store before decrementing the global barrier so other CTAs / the host see
// the write.
template<typename C2M_Type>
__device__ __forceinline__ void stwarp_execute_singlethread(
    C2M_Type &c2m, const MInst* slot_insts,
    const void *smem_base, const CUtensorMap *tma_descs, int *bars) {

  __stprint("[ST Warp][AMD] Start ST warp execution");

  int slot_mask = c2m.pop();
  while (slot_mask) {

    auto slot = extract(slot_mask);
    bool do_free = true;

    __stprint("Receive ST slot: slot=%d", slot);

    auto &inst = slot_insts[slot];
    uint16_t opcode = inst.opcode;

    switch(op(opcode)) {
      case op(OP_ALLOC_WB_TMA_STORE_1D): {
        __stprint("[AMD] 1D Store: size=%d", inst.size);
        const char* __restrict__ src =
            reinterpret_cast<const char*>(get_slot_address(smem_base, slot));
        char* __restrict__ dst =
            reinterpret_cast<char*>(inst.address);
        const size_t n = inst.size;
        // Make sure compute's LDS writes are visible to this lane before we
        // copy out.
        __threadfence_block();
        const uintptr_t srcaddr = reinterpret_cast<uintptr_t>(src);
        const uintptr_t dstaddr = reinterpret_cast<uintptr_t>(dst);
        if (((srcaddr | dstaddr) & 0xF) == 0 && (n & 0xF) == 0) {
          const uint4* __restrict__ s4 = reinterpret_cast<const uint4*>(src);
          uint4* __restrict__ d4 = reinterpret_cast<uint4*>(dst);
          const size_t n16 = n >> 4;
          #pragma unroll 1
          for (size_t i = 0; i < n16; ++i) d4[i] = s4[i];
        } else {
          #pragma unroll 1
          for (size_t i = 0; i < n; ++i) dst[i] = src[i];
        }
        break;
      }
      default:
        // Multi-D TMA stores / reduce-add are not implemented on AMD.
        // Don't free the slot — this is unrecoverable.
        __stprint("Unsupported ST opcode on AMD: slot=%d op=%d opcode=%04x",
                  slot, op(inst.opcode), inst.opcode);
        do_free = false;
        __builtin_trap();
        break;
    }

    if (opcode & MEM_OP_FLAGS_BARRIER) {
      // Make the global write visible across the device before the barrier
      // decrement. Hopper relies on cp.async.bulk's group-fence semantics;
      // here we issue an explicit device-scope fence.
      __threadfence();
      atomicSub(&bars[inst.bar()], 1);
    }

    __stprint("finish slot=%d op=%d flags=%02x",
      slot, op(inst.opcode), opcode & ((1 << flagBits) - 1));

    if (do_free)
      c2m.reset(slot_mask);
    slot_mask = c2m.pop();
  }

  __stprint("[AMD] End of ST warp execution");
}
#else

// TODO(zhiyuang): attach bars to the writeback
template<typename C2M_Type>
__device__ __forceinline__ void stwarp_execute_singlethread(
    C2M_Type &c2m, const MInst* slot_insts,
    const void *smem_base, const CUtensorMap *tma_descs, int *bars) {

  __stprint("[ST Warp] Start ST warp execution");
    
  int slot_mask = c2m.pop();
  while (slot_mask) {
  
    auto slot = extract(slot_mask);
    bool do_free = true;

    __stprint("Receive ST slot: slot=%d", slot);

    auto &inst = slot_insts[slot];
    uint16_t opcode = inst.opcode;
    // all ops are writeback ops

    switch(op(opcode)) {
      case op(OP_ALLOC_WB_TMA_STORE_1D):
      {
        cuda::ptx::cp_async_bulk(
          cuda::ptx::space_global,
          cuda::ptx::space_shared,
          (void*)(inst.address),
          (const void *)(get_slot_address(smem_base, slot)),
          inst.size
        );
        cuda::ptx::cp_async_bulk_commit_group();
      } 
        break;
      case op(OP_ALLOC_WB_TMA_STORE_2D):
        {
          const uint16_t *cord = inst.coords;
          __stprint("TMA 2D Store: desc_idx=%d size=%d cord=(%d,%d)",
                    inst.arg, inst.size, cord[0], cord[1]);
          asm volatile(
            "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
            "[%0, {%1, %2}], [%3];\n"
            :
            : "l"((void *)(tma_descs + inst.arg)),
              "r"((int)cord[0]),
              "r"((int)cord[1]),
              "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot)))
              : "memory");
          cuda::ptx::cp_async_bulk_commit_group();
        }
        break;
      case op(OP_ALLOC_WB_TMA_STORE_4D):
        {
          const uint16_t *cord = inst.coords;
          __stprint("TMA 4D Store: desc_idx=%d size=%d cord=(%d,%d,%d,%d)",
                    inst.arg, inst.size, cord[0], cord[1], cord[2], cord[3]);
          asm volatile(
            "cp.async.bulk.tensor.4d.global.shared::cta.bulk_group "
            "[%0, {%1, %2, %3, %4}], [%5];\n"
            :
            : "l"((void *)(tma_descs + inst.arg)),
              "r"((int)cord[0]),
              "r"((int)cord[1]),
              "r"((int)cord[2]),
              "r"((int)cord[3]),
              "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot)))
              : "memory");
          cuda::ptx::cp_async_bulk_commit_group();
        }
        break;
      case op(OP_ALLOC_WB_TMA_STORE_3D):
        {
          const uint16_t *cord = inst.coords;
          __stprint("TMA 3D Store: desc_idx=%d size=%d cord=(%d,%d,%d)",
                    inst.arg, inst.size, cord[0], cord[1], cord[2]);
          asm volatile(
            "cp.async.bulk.tensor.3d.global.shared::cta.bulk_group "
            "[%0, {%1, %2, %3}], [%4];\n"
            :
            : "l"((void *)(tma_descs + inst.arg)),
              "r"((int)cord[0]),
              "r"((int)cord[1]),
              "r"((int)cord[2]),
              "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot)))
              : "memory");
          cuda::ptx::cp_async_bulk_commit_group();
        }
        break;
      case op(OP_ALLOC_WB_TMA_STORE_5D_FIX0):
        {
          const uint16_t *cord = inst.coords;
          // harcode first coord to be 0
          __stprint("TMA 5D Store: desc_idx=%d size=%d cord=(0,%d,%d,%d,%d)",
                    inst.arg, inst.size, cord[0], cord[1], cord[2], cord[3]);
          asm volatile(
            "cp.async.bulk.tensor.5d.global.shared::cta.bulk_group "
            "[%0, {0, %1, %2, %3, %4}], [%5];\n"
            :
            : "l"((void *)(tma_descs + inst.arg)),
              "r"((int)cord[0]),
              "r"((int)cord[1]),
              "r"((int)cord[2]),
              "r"((int)cord[3]),
              "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot)))
              : "memory");
          cuda::ptx::cp_async_bulk_commit_group();
        }
        break;
      case op(OP_ALLOC_WB_TMA_REDUCE_ADD_2D):
        {
          const uint16_t *cord = inst.coords;
          __stprint("TMA 2D Reduce-Add: desc_idx=%d size=%d cord=(%d,%d)",
                    inst.arg, inst.size, cord[0], cord[1]);
          asm volatile(
            "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group "
            "[%0, {%1, %2}], [%3];\n"
            :
            : "l"((void *)(tma_descs + inst.arg)),
              "r"((int)cord[0]),
              "r"((int)cord[1]),
              "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot)))
              : "memory");
          cuda::ptx::cp_async_bulk_commit_group();
        }
        break;
      case op(OP_ALLOC_WB_TMA_REDUCE_ADD_3D):
        {
          const uint16_t *cord = inst.coords;
          __stprint("TMA 3D Reduce-Add: desc_idx=%d size=%d cord=(%d,%d,%d)",
                    inst.arg, inst.size, cord[0], cord[1], cord[2]);
          asm volatile(
            "cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.bulk_group "
            "[%0, {%1, %2, %3}], [%4];\n"
            :
            : "l"((void *)(tma_descs + inst.arg)),
              "r"((int)cord[0]),
              "r"((int)cord[1]),
              "r"((int)cord[2]),
              "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot)))
              : "memory");
          cuda::ptx::cp_async_bulk_commit_group();
        }
        break;
      default:
        // unknown opcode
        __stprint("Unknown mem wb opcode: slot_mask=%x slot=%d op=%d opcode=%04x\n", slot_mask, slot, op(inst.opcode), inst.opcode);
        do_free = false;
        break;
    }

    // do bar for all instructions all at once
    if (opcode & MEM_OP_FLAGS_BARRIER) {
      // cuda::std::atomic_ref<int> bar {bars[inst.bar()]};
      cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
      atomicSub(&bars[inst.bar()], 1);
      // int current_cnt = bar.fetch_sub(1, cuda::std::memory_order_release);
      // __stprint("Arrive for barrier %d, remaining count=%d", inst.bar(), current_cnt - 1);
      // if (inst.bar() == 0)
      //   printf("[sm=%d] Arrive for barrier %d, remaining count=%d\n", blockIdx.x, inst.bar(), current_cnt - 1);
    } else {
      cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<0>{});
    }

    // write back to free the slot
    __stprint("finish slot=%d op=%d flags=%02x",
      slot, op(inst.opcode), opcode & ((1 << flagBits) - 1));

    if (do_free)
      c2m.reset(slot_mask);
    slot_mask = c2m.pop();
  }

  __stprint("End of ST warp execution");
}

#endif  // __HIP_PLATFORM_AMD__
