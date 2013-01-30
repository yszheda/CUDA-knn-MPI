#include<stdio.h>
#include<pthread.h>
#include<stdlib.h>
#include<time.h>
#include<cuda.h>

#define INIT_MAX 10000000
#define TILE_WIDTH 32
#define TILE_DEPTH 128
#define MAX_BLOCK_SIZE 256
#define MAX_PTRNUM_IN_SMEM 1024 

// compute the square of distance of the ith point and jth point
__global__ void computeDist(int id, int m, int n, int *V, int *D)
{
	__shared__ int rowVector[TILE_WIDTH][TILE_DEPTH];
	__shared__ int colVector[TILE_DEPTH][TILE_WIDTH];
	__shared__ int dist[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;
   	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row;
	int col;
	int px;
	int py;	

	for(py=ty; py<TILE_WIDTH; py+=blockDim.y)
	{
		for(px=tx; px<TILE_WIDTH; px+=blockDim.x)
		{
			row = by*TILE_WIDTH+py;
			col = bx*TILE_WIDTH+px;
			dist[py][px] = 0;
			__syncthreads();
		
			if(row >= id*(m/2) && row < (id+1)*(m/2))
			{

				for(int i=0; i<(int)(ceil((float)n/TILE_DEPTH)); i++)
				{
					for(int j=tx; j<TILE_DEPTH; j+=blockDim.x)
					{
						rowVector[py][j] = V[row*n+i*TILE_DEPTH+j];
					}
					for(int j=ty; j<TILE_DEPTH; j+=blockDim.y)
					{		
						colVector[j][px] = V[col*n+i*TILE_DEPTH+j];
					}
					__syncthreads();
			
					for(int j=0; j<TILE_DEPTH; j++)
					{
						dist[py][px] += (rowVector[py][j]-colVector[j][px])*(rowVector[py][j]-colVector[j][px]);
					}
					__syncthreads();
				}
				
				if(row >= (m/2))
				{
					row -= (m/2);
				}

				D[row*m+col] = dist[py][px];
			}
		}
	}
}

extern __shared__ int SMem[];

//find the min value and index in the count^th loop
__device__ int findMin(int id, int m, int k, int count, int *D, int *out)
{
	int i = blockIdx.x;
  	int tid = threadIdx.x;

	int s = blockDim.x/2;
	int resultValue = INIT_MAX;
	int resultIndex = INIT_MAX;
	int indexBase = (m<MAX_PTRNUM_IN_SMEM)? m: MAX_PTRNUM_IN_SMEM;
	
	for(int num=0; num<m; num+=MAX_PTRNUM_IN_SMEM)
	{
		for(int j=tid; j<indexBase; j+=blockDim.x)
		{
			if(j+num == i+(m/2)*id )
			{
				SMem[j] = INIT_MAX;
			}
			else
			{
//				SMem[j] = D[(i + (m/2)*id) *m+num+j];
				SMem[j] = D[i*m+num+j];
			}
			//index
			SMem[indexBase+j] = j+num;
			__syncthreads();
		}
/*
		if(tid < count)
		{
			if(out[i*k+tid]-num>=0 && out[i*k+tid]-num<indexBase)
			{
				SMem[ out[i*k+tid]-num ] = INIT_MAX;
			}
			__syncthreads();
		}
		__syncthreads();
*/
		for(int j=0; j<count; j++)
		{
			if(out[i*k+j]-num>=0 && out[i*k+j]-num<indexBase)
			{
				SMem[ out[i*k+j]-num ] = INIT_MAX;
			}
			__syncthreads();
		}
		__syncthreads();
//		for(s=indexBase/2; s>0; s>>=1) 
		for(s=indexBase/2; s>32; s>>=1) 
		{
			for(int j=tid; j<indexBase; j+=blockDim.x)
			{
				if(j < s) 
				{
					if(SMem[j] == SMem[j+s])
					{
						if(SMem[indexBase+j] > SMem[indexBase+j+s])
						{
							SMem[indexBase+j] = SMem[indexBase+j+s];
						}
					}
					else if(SMem[j] > SMem[j+s])
					{
						SMem[j] = SMem[j+s];
						SMem[indexBase+j] = SMem[indexBase+j+s];
					}
				}
				__syncthreads();
			}
		}

		if(tid < 32)
		{
			#pragma unroll 5
			for(s=32; s>0; s>>=1)
			{ 
				if(SMem[tid] == SMem[tid+s])
				{
					if(SMem[indexBase+tid] > SMem[indexBase+tid+s])
					{
						SMem[indexBase+tid] = SMem[indexBase+tid+s];
					}
				}
				else if(SMem[tid] > SMem[tid+s])
				{
					SMem[tid] = SMem[tid+s];
					SMem[indexBase+tid] = SMem[indexBase+tid+s];
				}
			}
		}
	
		__syncthreads();
		if(resultValue == SMem[0])
		{
			if(resultIndex > SMem[indexBase])
			{
				resultIndex = SMem[indexBase];
			}
		} 
		else if (resultValue > SMem[0])
		{
			resultValue = SMem[0];
			resultIndex = SMem[indexBase];
		}
		__syncthreads();
	}
	return resultIndex;

}

// compute the k nearest neighbors
__global__ void knn(int id, int m, int k, int *V, int *D, int *out)
{
	int i;
	int count;

	i = blockIdx.x;
	__syncthreads();
	for(count=0; count<k; count++)
	{
		out[i*k+count] = findMin(id, m, k, count, D, out);
		__syncthreads();
	}
}

extern "C"
void beforeStart(const char* hostname){
	float *dA;
	// You can change processor_name according to your device settings.
	int processor_name = 0;
	while(1){
		cudaSetDevice(processor_name);
		if(cudaMalloc((void**)&dA, 1024*sizeof(float))){
			continue;
		}
		break;
	}
	cudaFree(dA);
	printf("you get device %d on %s\n", BDorNot, hostname);
}

extern "C"
void launch(int id, int m, int n, int k, int *V, int *out)
{
	int *d_V, *d_out;			// device copies
	int *D;						// distance matrix

	// allocate space for devices copies
	cudaMalloc((void **)&d_V, m*n*sizeof(int));
	cudaMalloc((void **)&D, (m/2)*m*sizeof(int));
	cudaMalloc((void **)&d_out, (m/2)*k*sizeof(int));
	
	// copy host values to devices copies
	cudaMemcpy(d_V, V, m*n*sizeof(int), cudaMemcpyHostToDevice);
	
	int gridDimX = (int)(ceil((float)m/TILE_WIDTH));
	int gridDimY = (int)(ceil((float)m/TILE_WIDTH));
	
	dim3 grid(gridDimX, gridDimY);
	dim3 block(TILE_WIDTH, TILE_WIDTH);
	
	// launch knn() kernel on GPU
	computeDist<<<grid, block>>>(id, m, n, d_V, D);
	cudaDeviceSynchronize();
	
	int threadNum = (m<MAX_BLOCK_SIZE)? m: MAX_BLOCK_SIZE;
	int ptrNumInSMEM = (m<MAX_PTRNUM_IN_SMEM)? m: MAX_PTRNUM_IN_SMEM;
	knn<<<(m/2), threadNum, 2*ptrNumInSMEM*sizeof(int)>>>(id, m, k, d_V, D, d_out);
	
	// copy result back to host
	cudaMemcpy(out+(m/2)*k*id, d_out, (m/2)*k*sizeof(int), cudaMemcpyDeviceToHost);
	
	// cleanup
	cudaFree(d_V);
	cudaFree(d_out);
	cudaFree(D);

}
