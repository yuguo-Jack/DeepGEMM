HIPCC ?= /opt/dtk/hip/bin/hipcc
AICC ?= /opt/dtk/bin/aicc
HOSTCXX ?= c++
DCCOBJDUMP ?= /opt/dtk/bin/dccobjdump

SRC := k1_groupgemm_v2.cpp
HIPCC_BIN := k1_groupgemm_v2_hipcc
AICC_BIN := k1_groupgemm_v2_aicc
PACK5_LAYOUT_CHECK_BIN := pack5_layout_check

INCLUDES := -I../../../deep_gemm/include
CXXFLAGS := -O3 -std=c++17 --offload-arch=gfx938 -DNDEBUG -mllvm -enable-num-vgprs-768=true $(INCLUDES)
AICCFLAGS := -x hip -O3 -std=c++17 --rocm-path=/opt/dtk --offload-arch=gfx938 -DNDEBUG -fno-finite-math-only -mllvm -enable-num-vgprs-768=true $(INCLUDES)
AICCLDFLAGS := -L/opt/dtk/lib -L/opt/dtk/hip/lib -Wl,-rpath,/opt/dtk/lib -Wl,-rpath,/opt/dtk/hip/lib

.PHONY: all hipcc aicc layout-check clean dump dump-hipcc

all: hipcc aicc

hipcc: $(HIPCC_BIN)

aicc: $(AICC_BIN)

layout-check: $(PACK5_LAYOUT_CHECK_BIN)

$(HIPCC_BIN): $(SRC)
	$(HIPCC) $(CXXFLAGS) $< -o $@

$(AICC_BIN): $(SRC)
	$(AICC) $(AICCFLAGS) $< -o $@ $(AICCLDFLAGS)

$(PACK5_LAYOUT_CHECK_BIN): pack5_layout_check.cpp
	$(HOSTCXX) -O2 -std=c++17 $< -o $@

dump: $(AICC_BIN)
	$(DCCOBJDUMP) --inputs=$(AICC_BIN) --show-sass --separate-functions > $(AICC_BIN).dump

dump-hipcc: $(HIPCC_BIN)
	$(DCCOBJDUMP) --inputs=$(HIPCC_BIN) --show-sass --separate-functions > $(HIPCC_BIN).dump

clean:
	rm -f $(HIPCC_BIN) $(AICC_BIN) $(PACK5_LAYOUT_CHECK_BIN) *.o *.s *.bc *.cui *.dump
