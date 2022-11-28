#include <cuda.h>
#define THREAD_PER_THREAD_BLOCK 256
// Create other necessary functions here

__global__ void dot_product(int* a, int*b, int*c, int N)
{
	int len=(N>>1)*(N>>1);
	int index= blockIdx.x*blockDim.x+threadIdx.x;
	if(index>=len)
	{
		return;
	}
	int i = index/(N>>1);
	int j = index%(N>>1);
	int rowA = i<<1;
	int colB = j<<1;
	int accumulator=0;
	int k=0;
	for(k=0;k<N;k++)
	{
		int temp=b[k*N+colB]+b[k*N+(colB+1)];
		int temp1=a[rowA*N+k]+a[(rowA+1)*N+k];
		accumulator+=temp*temp1;
	}

	c[index]=accumulator;
}

// Fill in this function
void gpuThread(int N, int *matA, int *matB, int *output)
{
	int* d_a, *d_b, *d_output;
	//you can think of transposing matrix B and then using the transposed matrix if you want
	if(cudaMalloc(&d_a, sizeof(int)*N*N)!=cudaSuccess)
	{
		cout<<"Error allocating memory for matrix A on GPU "<<endl ;
		exit(1);
	}
	if(cudaMalloc(&d_b, sizeof(int)*N*N)!=cudaSuccess)
	{
		cout<<"Error allocating memory for matrix B on GPU"<<endl;
		exit(1);
	}
	if(cudaMalloc(&d_output, sizeof(int)*(N>>1)*(N>>1))!=cudaSuccess)
	{
		cout<<"Error allocating memory for matrix output on GPU"<<endl;
		exit(1);
	}
	
	if(cudaMemcpy(d_a, matA, sizeof(int)*N*N, cudaMemcpyHostToDevice)!=cudaSuccess)
	{
		cout<<"Error copying matA to GPU "<<endl;
		exit(1);
	}
	if(cudaMemcpy(d_b, matB, sizeof(int)*N*N, cudaMemcpyHostToDevice)!=cudaSuccess)
	{
		cout<<"Error copying matB to GPU "<<endl;
		exit(1);
	}

	//call to the kernel

	int thread_blocks=((N>>1)*(N>>1)+THREAD_PER_THREAD_BLOCK-1)/THREAD_PER_THREAD_BLOCK;
	dot_product<<<thread_blocks, THREAD_PER_THREAD_BLOCK>>>(d_a, d_b, d_output, N);
	
	if(cudaMemcpy(output, d_output, sizeof(int)*(N>>1)*(N>>1), cudaMemcpyDeviceToHost)!=cudaSuccess)
	{
		cout<<"Error copying output matrix from GPU to host"<<endl;
		exit(1);
	}
}
