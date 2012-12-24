#include<stdio.h>
#include<pthread.h>
#include<stdlib.h>


__global__ void kernel(int *dA){
	int id = threadIdx.x;
	if(id+1 < 32) dA[id] += dA[id+1];
	if(id+2 < 32) dA[id] += dA[id+2];
	if(id+4 < 32) dA[id] += dA[id+4];
	if(id+8 < 32) dA[id] += dA[id+8];
	if(id+16 < 32) dA[id] += dA[id+16];
}

extern "C"
void launch(int *A){
	int *dA;
	cudaMalloc((void**)&dA, sizeof(int)*32);
	cudaMemcpy(dA, A, sizeof(int)*32, cudaMemcpyHostToDevice);
	kernel<<<1, 32>>>(dA);
	cudaMemcpy(A, dA, sizeof(int)*32, cudaMemcpyDeviceToHost);
	cudaFree(dA);
}
