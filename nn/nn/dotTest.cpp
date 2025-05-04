#include<iostream>
#include<ctime>
#include<thread>
#include"tensor.h"

using namespace Tensor;

void basicDot(const float* a, const float* b, float* c, const int n = 1000)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			for (int k = 0; k < n; k++)
			{
				c[i * n + j] += a[i * n + k] * b[k * n + j];
			}
		}
	}
}

void row_wise_dot(const float* a, const float* b, float* c, const int n = 1000)
{
	unsigned int cores = getNumCores();
	unsigned int rowBlocks = n / cores;
	unsigned int remainder = n % cores;

	std::vector<std::thread> threads;

	size_t start = 0;
	for (int i = 0; i < cores; i++)
	{
		size_t end = start + rowBlocks + (i < remainder ? 1 : 0);
		threads.emplace_back([start, end, n, a, b, c] {
			for (int i = start; i < end; i++)
			{
				for (int j = 0; j < n; j++)
				{
					for (int k = 0; k < n; k++)
					{
						c[i * n + j] += a[i * n + k] * b[k * n + j];
					}
				}
			}
			});
		start = end;
	}

	for (std::thread& t : threads) t.join();
}

void col_wise_dot(const float* a, const float* b, float* c, const int n = 1000)
{
	unsigned int cores = getNumCores();
	unsigned int colBlocks = n / cores;
	unsigned int remainder = n % cores;

	std::vector<std::thread> threads;

	size_t start = 0;
	for (int i = 0; i < cores; i++)
	{
		size_t end = start + colBlocks + (i < remainder ? 1 : 0);
		threads.emplace_back([start, end, n, a, b, c] {
			for (int j = start; j < end; j++)
			{
				for (int i = 0; i < n; i++)
				{
					for (int k = 0; k < n; k++)
					{
						c[i * n + j] += a[i * n + k] * b[k * n + j];
					}
				}
			}
			});
		start = end;
	}

	for (std::thread& t : threads) t.join();
}

void block_dot(const float* a, const float* b, float* c, const int n = 1000, const int blockSize = 64)
{
	unsigned int cores = getNumCores();
	std::vector<std::thread> threads;

	int numBlocks = (n + blockSize - 1) / blockSize;

	int totalBlockJobs = numBlocks * numBlocks;
	int jobsPerThread = totalBlockJobs / cores;
	int remainder = totalBlockJobs % cores;

	int startJob = 0;
	for (int i = 0; i < cores; ++i)
	{
		int endJob = startJob + jobsPerThread + (i < remainder ? 1 : 0);

		threads.emplace_back([=]() {
			for (int job = startJob; job < endJob; ++job)
			{
				int blockRow = job / numBlocks;
				int blockCol = job % numBlocks;

				int rowStart = blockRow * blockSize;
				int colStart = blockCol * blockSize;

				for (int kBlock = 0; kBlock < numBlocks; ++kBlock)
				{
					int kStart = kBlock * blockSize;

					for (int i = rowStart; i < std::min(rowStart + blockSize, n); ++i)
					{
						for (int j = colStart; j < std::min(colStart + blockSize, n); ++j)
						{
							for (int k = kStart; k < std::min(kStart + blockSize, n); ++k)
							{
								c[i * n + j] += a[i * n + k] * b[k * n + j];
							}
						}
					}
				}
			}
			});

		startJob = endJob;
	}

	for (auto& t : threads) t.join();
}

//int main(void)
//{
//	std::cout << getNumCores() << std::endl;
//	time_t start = clock();
//	const int n = 640;
//	float* a = new float[n*n]{  };
//	float* b = new float[n*n]{  };
//	float* c = new float[n*n]{};
//	/*for (int i = 0; i < 10; i++)
//	{
//		basicDot(a, b, c, n);
//	}*/
//	time_t end = clock();
//	//std::cout << "직렬 행렬곱 : " << end - start << std::endl;
//
//	start = clock();
//	for (int i = 0; i < 10; i++)
//	{
//		row_wise_dot(a, b, c, n);
//	}
//	end = clock();
//	std::cout << "row-wise 병렬 행렬곱 : " << end - start << std::endl;
//
//	start = clock();
//	for (int i = 0; i < 10; i++)
//	{
//		col_wise_dot(a, b, c, n);
//	}
//	end = clock();
//	std::cout << "col-wise 병렬 행렬곱 : " << end - start << std::endl;
//
//	start = clock();
//	for (int i = 0; i < 10; i++)
//	{
//		block_dot(a, b, c, n);
//	}
//	end = clock();
//	std::cout << "블록 병렬 행렬곱 : " << end - start << std::endl;
//}