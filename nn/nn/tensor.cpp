#include "tensor.h"
#include<assert.h>
#include<thread>

using namespace Tensor;

unsigned int getNumCores()
{
	static unsigned int numCores = [] {
		unsigned int temp = std::thread::hardware_concurrency();
		return temp ? temp : 4; }();
	return numCores;
}

std::ostream& operator<<(std::ostream out, Shape& s)
{
	out << "(" << s.shape[0];
	for (int i = 1; i < s.dimension; i++)
	{
		out << ", " << s.shape[i];
	}
	out << ")";
	return out;
}

bool Shape::operator==(Shape& s)
{
	if (dimension != s.dimension) return false;
	
	for (int i = 0; i < dimension; i++)
	{
		if (shape[i] != s.shape[i]) return false;
	}
	return true;
}

tensor::tensor(Shape& shape) : shape(shape)
{
	int size = 1;
	for (int i = 0; i < shape.dimension; i++)
	{
		size *= shape.shape[i];
	}
	this->arraySize = size;
	this->array = new float[arraySize] {};
}

int tensor::find_idx(const std::vector<int>& indices)
{
	assert(indices.size() == shape.dimension);
	int idx = 0;
	int stride = 1;
	for (int i = shape.dimension - 1; i >= 0; i--)
	{
		idx += indices[i] * stride;
		stride *= shape.shape[i];
	}
	return idx;
}

tensor& createFillTensor(Shape& shape, float value)
{
	int arraySize = 1;
	for (int i = 0; i < shape.dimension; i++)
	{
		arraySize *= shape.shape[i];
	}
	tensor* t =  new tensor(shape, new float[arraySize] {value});
	return *t;
}

std::ostream& operator<<(std::ostream& out, tensor& t)
{
	out << "tensor ";
}

tensor& tensor::operateFloat(float(*func)(float, float), float value)
{
	float* newArray = new float[arraySize] {};

	unsigned int numCores = getNumCores();
	size_t blockSize = arraySize / numCores;
	unsigned int remainder = arraySize % numCores;

	std::vector<std::thread> threads;
	size_t start = 0;

	for (int i = 0; i < numCores; i++)
	{
		size_t end = start + blockSize + (i < remainder ? 1 : 0);
		threads.emplace_back([start, end, value, newArray, func, this] {
			for (int idx = start; idx < end; idx++)
			{
				newArray[idx] = func(array[idx], value);
			}
			});
	}

	for (auto& t : threads) t.join();

	tensor* newTensor = new tensor(shape, newArray);
	return *newTensor;
}

tensor& tensor::operator+(const float a)
{
	return operateFloat([](float a, float b) {return a + b; }, a);
}

tensor& tensor::operator-(const float a)
{
	return operateFloat([](float a, float b) {return a - b; }, a);
}

tensor& tensor::operator*(const float a)
{
	return operateFloat([](float a, float b) {return a * b; }, a);
}

tensor& tensor::operator/(const float a)
{
	return operateFloat([](float a, float b) {return a / b; }, a);
}

tensor& tensor::operateTensor(float(*func)(float, float), const tensor& a)
{
	assert(this->shape == a.shape);
	float* newArray = new float[arraySize] {};

	unsigned int numCores = getNumCores();
	size_t blockSize = arraySize / numCores;
	unsigned int remainder = arraySize % numCores;

	std::vector<std::thread> threads;
	size_t start = 0;
	
	for (int i = 0; i < numCores; i++)
	{
		size_t end = start + blockSize + (i < remainder ? 1 : 0);
		threads.emplace_back([start, end, a, newArray, func, this] {
			for (int idx = start; idx < end; idx++)
			{
				newArray[idx] = func(array[idx], a.array[idx]);
			}
			});
	}

	for (auto& t : threads) t.join();

	tensor* newTensor = new tensor(shape, newArray);
	return *newTensor;
}

tensor& tensor::operator+(const tensor& a)
{
	return operateTensor([](float a, float b) {return a + b; }, a);
}

tensor& tensor::operator-(const tensor& a)
{
	return operateTensor([](float a, float b) {return a - b; }, a);
}

tensor& tensor::operator*(const tensor& a)
{
	return operateTensor([](float a, float b) {return a * b; }, a);
}

tensor& tensor::operator/(const tensor& a)
{
	return operateTensor([](float a, float b) {return a / b; }, a);
}