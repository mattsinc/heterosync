EXECUTABLE := allSyncPrims-1kernel
HIP_PATH ?= /opt/rocm/hip
SRC := $(wildcard *.hip.cpp)

all: release

release: $(SRC)
	$(HIP_PATH)/bin/hipcc -DGFX9 --amdgpu-target=gfx900 main.hip.cpp -o $(EXECUTABLE)

# gfx8 has a different number of bits it uses for sleeps, so compile accordingly
release-gfx8: $(SRC)
	$(HIP_PATH)/bin/hipcc --amdgpu-target=gfx803 main.hip.cpp -o $(EXECUTABLE)

debug: $(SRC)
	$(HIP_PATH)/bin/hipcc -DDEBUG -g -O0 main.hip.cpp -o $(EXECUTABLE).debug

clean:
	rm -f $(EXECUTABLE) $(EXECUTABLE).debug *~
