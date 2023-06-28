HIP_PATH ?= /opt/rocm/hip
SRC := $(wildcard *.cpp)
EXE = seqlocks

all: release

release: $(SRC)
	$(HIP_PATH)/bin/hipcc -DGFX9 --amdgpu-target=gfx900,gfx906 $(SRC) -o $(EXE)

# gfx8 has a different number of bits it uses for sleeps, so compile accordingly
release-gfx8: $(SRC)
	$(HIP_PATH)/bin/hipcc --amdgpu-target=gfx801,gfx803 $(SRC) -o $(EXE)

debug: $(SRC)
	$(HIP_PATH)/bin/hipcc -DDEBUG --amdgpu-target=gfx900,gfx906 -g -O0 $(SRC) -o $(EXE).debug

clean:
	rm -f $(EXE) $(EXE).debug *~
