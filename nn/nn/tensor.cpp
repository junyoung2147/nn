#include "tensor.h"
#include<assert.h>
#include<thread>

namespace Tensor
{

	unsigned int getNumCores()
	{
		static unsigned int numCores = [] {
			unsigned int temp = std::thread::hardware_concurrency();
			return temp ? temp : 4; }();
			return numCores;
	}

	std::ostream& operator<<(std::ostream& out, Shape& s)
	{
		out << "(" << s.dims[0];
		for (int i : s.dims)
		{
			out << ", " << i;
		}
		out << ")";
		return out;
	}

	bool Shape::operator==(const Shape& s)
	{
		if (dims.size() != s.dimension()) return false;

		for (int i = 0; i < dims.size(); i++)
		{
			if (dims[i] != s.dims[i]) return false;
		}
		return true;
	}

	tensor::tensor(Shape shape) : shape(shape)
	{
		int size = 1;
		for (int i : shape.dims)
		{
			size *= shape.dims[i];
		}
		//std::cout << size;
		this->arraySize = size;
		this->array = std::make_shared<float[]>(arraySize);
	}

	tensor::tensor(Shape shape, std::shared_ptr<float[]> array)
	{
		this->shape = shape;
		int size = 1;
		for (int i : this->shape.dims)
		{
			size *= this->shape.dims[i];
		}
		this->arraySize = size;
		this->array = array;
	}

	tensor::tensor(std::vector<unsigned int> v)  
	{
		shape = { v };
		int size = 1;
		for (int i : shape.dims)
		{
			size *= shape.dims[i];
		}
		//std::cout << size;
		this->arraySize = size;
		this->array = std::make_shared<float[]>(arraySize);
	}

	/*tensor::tensor(const tensor& other)
	{

	}

	tensor::tensor(tensor&& other)
	{
		shape = other.shape;
		array = other.array;
		other.array = nullptr;
		arraySize = other.arraySize;
	}*/

	/*tensor& tensor::operator=(tensor&& other)
	{
		if (this != &other)
		{
			delete[] array;
			array = other.array;
			other.array = nullptr;
			shape = other.shape;
			arraySize = other.arraySize;
		}
		return *this;
	}*/

	int tensor::find_idx(const std::vector<int>& indices)
	{
		assert(indices.size() == shape.dimension());
		int idx = 0;
		int stride = 1;
		for (int i = shape.dimension() - 1; i >= 0; i--)
		{
			idx += indices[i] * stride;
			stride *= shape.dims[i];
		}
		return idx;
	}

	tensor Tensor::createFillTensor(Shape shape, float value)
	{
		int arraySize = 1;
		for (int i = 0; i < shape.dimension(); i++)
		{
			arraySize *= shape.dims[i];
		}
		std::shared_ptr<float[]> array = std::make_shared<float[]>(arraySize);
		std::fill(array.get(), array.get() + arraySize, value);
		tensor* t = new tensor(shape, array);
		return *t;
	}

	tensor tensor::operateFloat(float(*func)(float, float), float value)
	{
		std::shared_ptr<float[]> newArray = std::make_shared<float[]>(arraySize);

		unsigned int numCores = getNumCores();
		size_t blockSize = arraySize / numCores;
		unsigned int remainder = arraySize % numCores;

		std::vector<std::thread> threads;
		size_t start = 0;

		for (unsigned int i = 0; i < numCores; i++)
		{
			//논리 코어 수로 나눈 나머지를 배분
			size_t end = start + blockSize + (i < remainder ? 1 : 0);
			//람다를 이용해 각 스레드에 start - end 범위의 작업을 맡김
			threads.emplace_back([start, end, value, newArray, func, this] {
				for (size_t idx = start; idx < end; idx++)
				{
					//인수로 받은 사칙연산 함수를 적용
					newArray[idx] = func(array[idx], value);
				}
				});
			start = end;
		}

		for (auto& t : threads) t.join();

		return tensor(shape, newArray);
	}

	tensor tensor::operator+(const float a)
	{
		return operateFloat([](float a, float b) {return a + b; }, a);
	}

	tensor tensor::operator-(const float a)
	{
		return operateFloat([](float a, float b) {return a - b; }, a);
	}

	tensor tensor::operator*(const float a)
	{
		return operateFloat([](float a, float b) {return a * b; }, a);
	}

	tensor tensor::operator/(const float a)
	{
		return operateFloat([](float a, float b) {return a / b; }, a);
	}

	tensor tensor::operateTensor(float(*func)(float, float), const tensor& a)
	{
		assert(this->shape == a.shape);
		std::shared_ptr<float[]> newArray = std::make_shared<float[]>(arraySize);

		unsigned int numCores = getNumCores();
		size_t blockSize = arraySize / numCores;
		unsigned int remainder = arraySize % numCores;

		std::vector<std::thread> threads;
		size_t start = 0;

		for (unsigned int i = 0; i < numCores; i++)
		{
			size_t end = start + blockSize + (i < remainder ? 1 : 0);
			threads.emplace_back([start, end, a, newArray, func, this] {
				for (size_t idx = start; idx < end; idx++)
				{
					newArray[idx] = func(array[idx], a.array[idx]);
				}
				});
			start = end;
		}

		for (auto& t : threads) t.join();

		return tensor(shape, newArray);
	}

	tensor tensor::operator+(const tensor& a)
	{
		return operateTensor([](float a, float b) {return a + b; }, a);
	}

	tensor tensor::operator-(const tensor& a)
	{
		return operateTensor([](float a, float b) {return a - b; }, a);
	}

	tensor tensor::operator*(const tensor& a)
	{
		return operateTensor([](float a, float b) {return a * b; }, a);
	}

	tensor tensor::operator/(const tensor& a)
	{
		return operateTensor([](float a, float b) {return a / b; }, a);
	}

	tensor tensor::dot(const tensor& a)
	{
		//row를 단위로 병렬 처리, 3차원 이상인 경우 상위 2차원을 기준으로 행렬곱
		unsigned int m1_d = shape.dimension();
		unsigned int m2_d = a.shape.dimension();
		assert(m1_d > 1 && m2_d > 1 && m1_d == m2_d);
		assert(shape.dims[m1_d - 1] == a.shape.dims[m2_d - 2]);
		//m1[i, k] X m2[k, j] 에서의 i, j, k 
		unsigned int rows = shape.dims[m1_d - 2];
		unsigned int cols = a.shape.dims[m1_d - 1];
		unsigned int cPart = shape.dims[m1_d - 1];
		std::cout << arraySize << ", " << cPart << std::endl;
		unsigned int totalRows = arraySize / cPart;
		std::shared_ptr<float[]> newArray = std::make_shared<float[]>(totalRows * cols);

		unsigned int cores = std::min(getNumCores(), totalRows);
		std::cout << cores;
		std::vector<std::thread> threads;
		unsigned int blocks = totalRows / cores;
		unsigned int remainder = totalRows % cores;

		for (unsigned int t = 0, start = 0; t < cores; ++t) {
			unsigned int end = start + blocks + (t < remainder ? 1 : 0);
			threads.emplace_back([start, end, rows, cols, cPart, this, newArray, &a]() {
				for (unsigned int idx = start; idx < end; ++idx) {
					//현재 idx 이전의 행렬 수
					unsigned int batch = idx / rows;
					//현재 행
					unsigned int i = idx % rows;
					for (unsigned int j = 0; j < cols; ++j) {
						float sum = 0;
						for (unsigned int k = 0; k < cPart; ++k) {
							//행렬 크기(행 x 열) * 현재 idx 이전의 행렬 개수 + 행 * 전체 열 + 열 로 인덱스 계산
							sum += array[batch * rows * cPart + i * cPart + k] *
								a.array[batch * cPart * cols + k * cols + j];
						}
						newArray[batch * rows * cols + i * cols + j] = sum;
					}
				}
				});
			start = end;
		}
		for (auto& t : threads) t.join();

		std::vector<unsigned int> shapeArray(shape.dims);
		shapeArray[m1_d - 1] = cols;
		Shape newShape = { shapeArray };
		return tensor(newShape, newArray);
	}

	void tensor::reshape(Shape& shape)
	{
		int size = 1;
		for (int i : shape.dims)
		{
			size *= i;
		}
		if (size != arraySize) return;
		this->shape = shape;
	}

	Shape tensor::getShape()
	{
		return shape;
	}

	std::ostream& operator<<(std::ostream& out, tensor& t)
	{
		out << "tensor ";
		tensorPrint(0, 0, t.shape.dims, t.array, out);
		return out;
	}

	void tensorPrint(unsigned int dim, unsigned int idx, std::vector<unsigned int>& dims, std::shared_ptr<float[]> array, std::ostream& out)
	{
		if (dim < dims.size() - 1) {
			unsigned int blockSize = 1;
			for (unsigned int j = dim + 1; j < dims.size(); j++)
			{
				blockSize *= dims[j];
			}

			for (unsigned int i = 0; i < dims[dim]; i++)
			{
				out << "[";
				tensorPrint(dim + 1, idx + blockSize*i, dims, array, out);
				out << "]" ;
			}
		}
		else
		{
			//std::cout << "last dim" << std::endl;
			out << array[idx];
			for (int i = 1; i < dims[dim]; i++)
			{
				//std::cout << idx + i;
				out << "," << array[idx + i];
			}
		}
	}
}