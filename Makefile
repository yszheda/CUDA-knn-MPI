
CC = mpicc
CXX = mpic++
CCC = mpic++
F77 = mpif77
FC = mpif90

CFLAGS = -g -lcudart -L/usr/local/cuda/lib64 -lrt
CXXFLAGS = -g
CCFLAGS = -g
F77FLAGS = -g
FCFLAGS = -g

MAIN = main

# Please fill the execution path of your program here:
EXEC_PATH = 

# Please put hostnames in a file and set the file name in HOST_FILE:
HOST_FILE =            

# Default target.  Always build the C example.  Only build the others
# if Open MPI was build with the relevant language bindings.

all: $(MAIN)

$(MAIN): $(MAIN).c kernel.o

kernel.o: kernel.cu
	nvcc -o kernel.o -c kernel.cu -lpthread -arch=sm_20

run: $(MAIN)
	/usr/local/bin/mpirun -np 2 -hostfile $(HOST_FILE) --mca btl_tcp_if_include eth0 --mca orte_default_hostname $(HOST_FILE) $(EXEC_PATH)/main $(IN) > $(OUT)

clean:
	rm -f $(MAIN) *~ *.o

