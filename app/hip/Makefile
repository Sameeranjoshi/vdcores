HIPCC    ?= hipcc
ARCH     ?= gfx942        # MI300X (CDNA3)
CXXFLAGS ?= -O2 -std=c++17 --offload-arch=$(ARCH) -D__AMDGCN_WAVEFRONT_SIZE=64

TARGET := vdcores_hip_demo
SRC    := vdcores_hip_demo.cpp

$(TARGET): $(SRC)
	$(HIPCC) $(CXXFLAGS) -o $@ $<

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: run clean
