
CC = mpicc
CXX = mpic++
CCC = mpic++
F77 = mpif77
FC = mpif90

CFLAGS = -g -lcudart -L/usr/local/cuda/lib64
CXXFLAGS = -g
CCFLAGS = -g
F77FLAGS = -g
FCFLAGS = -g

MAIN = main
EXECPATH = ~/hw3/

# Default target.  Always build the C example.  Only build the others
# if Open MPI was build with the relevant language bindings.

all: $(MAIN)

$(MAIN): $(MAIN).c knn.o

knn.o: knn.cu
	nvcc -o knn.o -c knn.cu -lpthread -arch=sm_20
#run: $(MAIN)
#	/usr/local/bin/mpirun -np 3 -hostfile $(EXECPATH)/hostfile --mca btl_tcp_if_include eth0 $(EXECPATH)/main
sub: job.sh main
	qsub job.sh
clean:
	rm -f $(MAIN) *~ *.o


