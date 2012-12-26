
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
EXEC_PATH = "/home/hcap/101062468/knn-MPI"
           

# Default target.  Always build the C example.  Only build the others
# if Open MPI was build with the relevant language bindings.

all: $(MAIN)

$(MAIN): $(MAIN).c kernel.o

kernel.o: kernel.cu
	nvcc -o kernel.o -c kernel.cu -lpthread -arch=sm_20

#change the hostfile according to your group
run: $(MAIN)
	/usr/local/bin/mpirun -np 2 -hostfile $(EXEC_PATH)/hostfileAB --mca btl_tcp_if_include eth0 --mca orte_default_hostname $(EXEC_PATH)/hostfileAB $(EXEC_PATH)/main $(EXEC_PATH)/test_large.txt > $(EXEC_PATH)/test_large.out-t
clean:
	rm -f $(MAIN) *~ *.o

