HIPCC    ?= hipcc
# Auto-detect the local GPU arch via rocminfo; fall back to gfx942 (MI300X) if
# unavailable. Override explicitly with `make ARCH=gfx90a` for MI210, etc.
ARCH     ?= $(shell rocminfo 2>/dev/null | awk '/Name:[[:space:]]*gfx/ {print $$2; exit}')
ifeq ($(strip $(ARCH)),)
ARCH := gfx942
endif
CXXFLAGS ?= -O2 -std=c++17 --offload-arch=$(ARCH) -D__AMDGCN_WAVEFRONT_SIZE=64

TARGETS := vdcores_hip_demo vdcores_hip_l1 vdcores_hip_l2 vdcores_hip_l3 dae_copy_smoke

all: $(TARGETS)

vdcores_hip_demo: vdcores_hip_demo.cpp
	$(HIPCC) $(CXXFLAGS) -o $@ $<

vdcores_hip_l1: vdcores_hip_l1.cpp
	$(HIPCC) $(CXXFLAGS) -o $@ $<

vdcores_hip_l2: vdcores_hip_l2.cpp
	$(HIPCC) $(CXXFLAGS) -o $@ $<

vdcores_hip_l3: vdcores_hip_l3.cpp
	$(HIPCC) $(CXXFLAGS) -o $@ $<

dae_copy_smoke: dae_copy_smoke.cpp
	$(HIPCC) $(CXXFLAGS) -o $@ $<

run-l0: vdcores_hip_demo
	./vdcores_hip_demo

run-l1: vdcores_hip_l1
	./vdcores_hip_l1

run-l2: vdcores_hip_l2
	./vdcores_hip_l2

run-l3: vdcores_hip_l3
	./vdcores_hip_l3

run-smoke: dae_copy_smoke
	./dae_copy_smoke

clean:
	rm -f $(TARGETS)

.PHONY: all run-l0 run-l1 run-l2 run-l3 run-smoke clean
