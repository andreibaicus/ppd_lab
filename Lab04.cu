#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <device_launch_parameters.h>

constexpr auto PI = 3.14f;

__global__ void array2D(float *a, float *b, int N, int M)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < N && col < M)
	{
		a[row*N + col] = powf(sinf(2 * PI * row / N), 2) + powf(cosf(2 * PI * col / N), 2);
		b[row*N + col] = powf(cosf(2 * PI * row / N), 2) + powf(sinf(2 * PI * col / N), 2);
	}
}

__global__ void array1D(float *a, float *b, int N, int M)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int row = i / N;
	int col = i % M;

	if (row < N && col < M)
	{
		a[row*N + col] = powf(sinf(2 * PI * row / N), 2) + powf(cosf(2 * PI * col / N), 2);
		b[row*N + col] = powf(cosf(2 * PI * row / N), 2) + powf(sinf(2 * PI * col / N), 2);
	}
}

__global__ void sum2D(float *a, float *b, float *c, int N, int M)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < N && col < M)
	{
		c[row * N + col] = a[row * N + col] + b[row * N + col];
	}
}

__global__ void sum1D(float *a, float *b, float *c, int N, int M)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int row = i / N;
	int col = i % M;

	if (row < N && col < M)
	{
		c[row * N + col] = a[row * N + col] + b[row * N + col];
	}
}

int main()
{
	float *a_h, *b_h, *c_h, *a_d, *b_d, *c_d;
	int N = 512;
	int M = 512;

	//alocare memorie host
	a_h = (float*)malloc(N * M * sizeof(float));
	b_h = (float*)malloc(N * M * sizeof(float));
	c_h = (float*)malloc(N * M * sizeof(float));

	//alocare device
	cudaMalloc((void**)&a_d, N * M * sizeof(float));
	cudaMalloc((void**)&b_d, N * M * sizeof(float));
	cudaMalloc((void**)&c_d, N * M * sizeof(float));

	//dimensiunea grid,bloc
	dim3 grid2D(16, 16, 1);
	dim3 threads2D(32, 32, 1);
	dim3 grid1D(512, 1, 1);
	dim3 threads1D(512, 1, 1);

	//apelare kerner
	//array2D << <grid2D, threads2D >> > (a_d, b_d, N, M);
	//sum2D << <grid2D, threads2D >> > (a_d, b_d, c_d, N, M);
	array1D << <grid1D, threads1D >> > (a_d, b_d, N, M);
	sum1D << <grid1D, threads1D >> > (a_d, b_d, c_d, N, M);

	//copiere gpu to cpu
	cudaMemcpy(a_h, a_d, N * M * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(b_h, b_d, N * M * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(c_h, c_d, N * M * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < M; ++j)
		{
			std::cout << c_h[i*N + j] << " ";
		}
		std::cout << std::endl;
	}

	//eliberare memorie
	free(a_h);
	free(b_h);
	cudaFree(a_d);
	cudaFree(b_d);

	return 0;
}