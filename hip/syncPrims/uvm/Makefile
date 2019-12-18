EXECUTABLE := allSyncPrims-1kernel.hip
HIP_PATH ?= /opt/rocm/hip
SRC := $(wildcard *.hip.cpp)

all: release

release: $(SRC)
	$(HIP_PATH)/bin/hipcc main.hip.cpp -o $(EXECUTABLE)

debug: $(SRC)
	$(HIP_PATH)/bin/hipcc -DDEBUG -g -O0 main.hip.cpp -o $(EXECUTABLE).debug

clean:
	rm -f $(EXECUTABLE) $(EXECUTABLE).debug *~
