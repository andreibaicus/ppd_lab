#include <iostream> 
#include <cuda.h>
#include<cuda_runtime.h>
#include <time.h>
#include <stdio.h>
using namespace std;

__global__ void initKernel(float *a, float *b, int N, int M) {

	__shared__ float img_copy[32][32];

	int row = blockIdx.x*blockDim.x + threadIdx.x;
	int col = blockIdx.y*blockDim.y + threadIdx.y;


	img_copy[row][col] = a[row*N + col];

	__syncthreads();
	float sum = 0;
	if (row == 0 || row == N - 1 || col == 0 || col == N - 1)
		return;
	for (int m = row - 1; m <= row + 1; m++)
		for (int n = col - 1; n <= col + 1; n++)
		{
			sum += img_copy[m][n];
		}
	sum /= 9;
	b[row*N + col] = sum;
	//b[row*N+col] = img_copy[row][col];

}



int main()
{

	int N = 32;
	int M = 32;
	dim3 nThreadsPerBlock(32, 32, 1);
	dim3 nBlocks(N / 32, M / 32, 1);


	float *a, *b, *img_copy;
	a = new float[N*M];
	b = new float[N*M];



	// Initializare matrice
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			//a[i*N+j]=(i+j)%2;
			a[i*N + j] = 1;
		}
	}

	float *a_d, *b_d;
	//      a_d = new float[N*M];
	//      b_d = new float[N*M];

	cudaMalloc((void**)&a_d, N*M * sizeof(float));
	cudaMalloc((void**)&b_d, N*M * sizeof(float));




	cudaMemcpy(a_d, a, N*M * sizeof(float), cudaMemcpyHostToDevice);
	initKernel << <nBlocks, nThreadsPerBlock >> > (a_d, b_d, N, M);
	cudaMemcpy(b, b_d, N*M * sizeof(float), cudaMemcpyDeviceToHost);


	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			cout << b[N*i + j] << " ";

		}
		cout << "\n";
	}
	return 0;
}