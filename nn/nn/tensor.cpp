#include "tensor.h"
#include<assert.h>
#include<thread>
#include<cmath>
#include<random>
#include<atomic>

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
		for (int i = 1; i <  s.dimension(); i++)
		{
			out << ", " << s.dims[i];
		}
		out << ")";
		return out;
	}

	bool Shape::operator==(const Shape& s) const
	{
		if (dims.size() != s.dimension()) return false;

		for (int i = 0; i < dims.size(); i++)
		{
			if (dims[i] != s.dims[i]) return false;
		}
		return true;
	}

	tensor::tensor()
	{
		shape = {};
		stride = {};
		offset = 0;
		arraySize = 0;
		array = nullptr;
	}

	tensor::tensor(Shape shape) : shape(shape)
	{
		offset = 0;
		int size = 1;
		for (int i : shape.dims)
		{
			size *= i;
		}
		//std::cout << size;
		this->arraySize = size;
		this->array = std::make_shared<float[]>(arraySize);
		initStride();
	}

	tensor::tensor(Shape shape, std::shared_ptr<float[]> array)
	{
		offset = 0;
		this->shape = shape;
		int size = 1;
		for (int i : this->shape.dims)
		{
			size *= i;
		}
		this->arraySize = size;
		this->array = array;
		initStride();
	}

	tensor::tensor(std::vector<unsigned int> v)  
	{
		//std::cout << "tensor constructor" << std::endl;
		offset = 0;
		shape = { v };
		//std::cout << shape << std::endl;
		int size = 1;
		for (unsigned int i : shape.dims)
		{
			size *= i;
		}
		//std::cout << size;
		this->arraySize = size;
		this->array = std::make_shared<float[]>(arraySize);
		initStride();
	}

	void tensor::initStride()
	{
		int size = shape.dimension();
		stride = std::vector<unsigned int>(size);
		stride[size - 1] = 1;
		for (int i = size - 2; i >= 0; i--)
		{
			stride[i] = stride[i + 1] * shape.dims[i + 1];
		}
	}

	bool tensor::is_contiguous() const {
		unsigned int expected_stride = 1;
		for (int i = shape.dimension() - 1; i >= 0; --i) {
			if (stride[i] != expected_stride)
				return false;
			expected_stride *= shape.dims[i];
		}
		return true;
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

	int tensor::find_idx(const std::vector<int>& indices) const
	{
		assert(indices.size() == shape.dimension());
		int idx = offset;
		//int stride = 1;
		for (int i = shape.dimension() - 1; i >= 0; i--)
		{
			/*idx += indices[i] * stride;
			stride *= shape.dims[i];*/
			idx += indices[i] * stride[i];
		}
		return idx;
	}

	void tensor::get_multi_idx(std::vector<int>& multi_idx, unsigned int idx) const
	{
		for (int i = shape.dimension() - 1; i >= 0; i--)
		{
			multi_idx[i] = idx % shape.dims[i];
			idx /= shape.dims[i];
		}
	}

	tensor Tensor::initFillTensor(Shape shape, float value)
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

	tensor tensor::operateFloat(float(*func)(float, float), float value) const
	{
		std::shared_ptr<float[]> newArray = std::make_shared<float[]>(arraySize);

		unsigned int numCores = getNumCores();
		size_t blockSize = arraySize / numCores;
		unsigned int remainder = arraySize % numCores;

		std::vector<std::thread> threads;
		size_t start = 0;

		if (is_contiguous())
		{
			for (unsigned int i = 0; i < numCores; i++)
			{
				//논리 코어 수로 나눈 나머지를 배분
				size_t end = start + blockSize + (i < remainder ? 1 : 0);
				//람다를 이용해 각 스레드에 start - end 범위의 작업을 맡김
				threads.emplace_back([start, end, value, newArray, func, this] {
					for (size_t idx = start; idx < end; idx++)
					{
						//인수로 받은 사칙연산 함수를 적용
						newArray[idx] = func(array[offset + idx], value);
					}
					});
				start = end;
			}
		}
		else
		{
			for (unsigned int i = 0; i < numCores; i++)
			{
				size_t end = start + blockSize + (i < remainder ? 1 : 0);
				threads.emplace_back([start, end, value, newArray, func, this] {
					std::vector<int> multi_idx(shape.dimension(), 0);
					for (size_t idx = start; idx < end; idx++)
					{
						get_multi_idx(multi_idx, idx);
						newArray[idx] = func(array[find_idx(multi_idx)], value);
					}
					});
				start = end;
			}
		}

		for (auto& t : threads) t.join();

		return tensor(shape, newArray);
	}

	tensor tensor::operator+(const float a) const
	{
		return operateFloat([](float a, float b) {return a + b; }, a);
	}

	tensor tensor::operator-(const float a) const
	{
		return operateFloat([](float a, float b) {return a - b; }, a);
	}

	tensor operator-(const float a, const tensor& b)
	{
		return b.operateFloat([](float a, float b) {return b - a; }, a);
	}

	tensor tensor::operator*(const float a) const
	{
		return operateFloat([](float a, float b) {return a * b; }, a);
	}

	tensor tensor::operator/(const float a) const
	{
		return operateFloat([](float a, float b) {return a / b; }, a);
	}

	tensor operator/(const float a, const tensor& b)
	{
		return b.operateFloat([](float a, float b) {return b / a; }, a);
	}

	tensor tensor::operateTensor(float(*func)(float, float), const tensor& a) const
	{
		assert(this->shape == a.shape);
		std::shared_ptr<float[]> newArray = std::make_shared<float[]>(arraySize);

		unsigned int numCores = getNumCores();
		size_t blockSize = arraySize / numCores;
		unsigned int remainder = arraySize % numCores;

		std::vector<std::thread> threads;
		size_t start = 0;

		if (is_contiguous() && a.is_contiguous())
		{
			for (unsigned int i = 0; i < numCores; i++)
			{
				//논리 코어 수로 나눈 나머지를 배분
				size_t end = start + blockSize + (i < remainder ? 1 : 0);
				//람다를 이용해 각 스레드에 start - end 범위의 작업을 맡김
				threads.emplace_back([start, end, a, newArray, func, this] {
					for (size_t idx = start; idx < end; idx++)
					{
						//인수로 받은 사칙연산 함수를 적용
						newArray[idx] = func(array[offset + idx], a.array[a.offset + idx]);
					}
					});
				start = end;
			}
		}
		else
		{
			for (unsigned int i = 0; i < numCores; i++)
			{
				size_t end = start + blockSize + (i < remainder ? 1 : 0);
				threads.emplace_back([start, end, a, newArray, func, this] {
					std::vector<int> multi_idx(shape.dimension(), 0);
					for (size_t idx = start; idx < end; idx++)
					{
						get_multi_idx(multi_idx, idx);
						newArray[idx] = func(array[find_idx(multi_idx)], a.array[a.find_idx(multi_idx)]);
					}
					});
				start = end;
			}
		}

		for (auto& t : threads) t.join();

		return tensor(shape, newArray);
	}

	tensor tensor::operator+(const tensor& a) const
	{
		return operateTensor([](float a, float b) {return a + b; }, a);
	}

	tensor tensor::operator-(const tensor& a) const
	{
		return operateTensor([](float a, float b) {return a - b; }, a);
	}

	tensor tensor::operator*(const tensor& a) const
	{
		return operateTensor([](float a, float b) {return a * b; }, a);
	}

	tensor tensor::operator/(const tensor& a) const
	{
		return operateTensor([](float a, float b) {return a / b; }, a);
	}

	tensor tensor::operator<(const tensor& other) const
	{
		return operateTensor([](float a, float b)->float {return a < b; }, other);
	}

	tensor tensor::operator>(const tensor& other) const
	{
		return operateTensor([](float a, float b)->float {return a > b; }, other);
	}

	tensor tensor::operator==(const tensor& other) const
	{
		return operateTensor([](float a, float b)->float {return a == b; }, other);
	}

	tensor tensor::operator<(const float other) const
	{
		return operateFloat([](float a, float b)->float {return a < b; }, other);
	}

	tensor tensor::operator>(const float other) const
	{
		return operateFloat([](float a, float b)->float {return a > b; }, other);
	}

	tensor tensor::operator==(const float other) const
	{
		return operateFloat([](float a, float b)->float {return a == b; }, other);
	}

	void tensor::updateTensor(float(*func)(float, float), const tensor& a) const
	{
		assert(this->shape == a.shape);
		unsigned int numCores = std::min(getNumCores(), (unsigned int)arraySize);
		unsigned int blockSize = arraySize / numCores;
		unsigned int remainder = arraySize % numCores;

		std::vector<std::thread> threads;
		unsigned int start = 0;

		if (is_contiguous() && a.is_contiguous())
		{
			//std::cout << "contiguous" << std::endl;
			for (unsigned short i = 0; i < numCores; i++)
			{
				size_t end = start + blockSize + (i < remainder ? 1 : 0);
				threads.emplace_back([start, end, a, func, this] {
					for (unsigned int idx = start; idx < end; idx++)
					{
						this->array[offset + idx] = func(array[offset + idx], a.array[a.offset + idx]);
					}
					});
				start = end;
			}
		}
		else
		{
			for (unsigned short i = 0; i < numCores; i++)
			{
				size_t end = start + blockSize + (i < remainder ? 1 : 0);
				threads.emplace_back([start, end, a, func, this] {
					std::vector<int> multi_idx(shape.dimension(), 0);
					for (unsigned int idx = start; idx < end; idx++)
					{
						get_multi_idx(multi_idx, idx);
						this->array[offset + idx] = func(array[find_idx(multi_idx)], a.array[a.find_idx(multi_idx)]);
					}
					});
				start = end;
			}
		}

		for (auto& t : threads) t.join();
	}

	tensor& tensor::operator+=(const tensor& a)
	{
		assert(this->shape == a.shape);
		updateTensor([](float a, float b) {return a + b; }, a);
		return *this;
	}

	tensor& tensor::operator-=(const tensor& a)
	{
		assert(this->shape == a.shape);
		updateTensor([](float a, float b) {return a - b; }, a);
		return *this;
	}

	void tensor::updateFloat(float(*func)(float, float), const float a) const
	{
		unsigned int numCores = std::min(getNumCores(), (unsigned int)arraySize);
		unsigned int blockSize = arraySize / numCores;
		unsigned int remainder = arraySize % numCores;

		std::vector<std::thread> threads;
		unsigned int start = 0;

		if (is_contiguous())
		{
			for (unsigned short i = 0; i < numCores; i++)
			{
				size_t end = start + blockSize + (i < remainder ? 1 : 0);
				threads.emplace_back([start, end, a, func, this] {
					for (unsigned int idx = start; idx < end; idx++)
					{
						this->array[offset + idx] = func(array[offset + idx], a);
					}
					});
				start = end;
			}
		}
		else
		{
			for (unsigned short i = 0; i < numCores; i++)
			{
				size_t end = start + blockSize + (i < remainder ? 1 : 0);
				threads.emplace_back([start, end, a, func, this] {
					std::vector<int> multi_idx(shape.dimension(), 0);
					for (unsigned int idx = start; idx < end; idx++)
					{
						get_multi_idx(multi_idx, idx);
						this->array[offset + idx] = func(array[find_idx(multi_idx)], a);
					}
					});
				start = end;
			}
		}

		for (auto& t : threads) t.join();
	}

	tensor& tensor::operator+=(const float a)
	{
		updateFloat([](float a, float b) {return a + b; }, a);
		return *this;
	}

	tensor& tensor::operator-=(const float a)
	{
		updateFloat([](float a, float b) {return a - b; }, a);
		return *this;
	}

	tensor tensor::operateBroadcast(tensor(*func)(const tensor&, const tensor&), const tensor& a, const unsigned int dim) const
	{
		assert(shape.dimension() == a.shape.dimension());
		assert(a.shape.dims[dim] == 1);
		tensor brod_tensor = a;
		brod_tensor.stride[dim] = 0;
		brod_tensor.shape.dims[dim] = shape.dims[dim];
		return func(*this, brod_tensor);
	}

	tensor tensor::broadcast_add(const tensor& a, const unsigned int dim) const
	{
		return operateBroadcast([](const tensor& a, const tensor& b) {
			return a + b;
			}, a, dim);
	}

	tensor tensor::broadcast_sub(const tensor& a, const unsigned int dim) const
	{
		return operateBroadcast([](const tensor& a, const tensor& b) {
			return a - b;
			}, a, dim);
	}

	tensor tensor::broadcast_mul(const tensor& a, const unsigned int dim) const
	{
		return operateBroadcast([](const tensor& a, const tensor& b) {
			return a * b;
			}, a, dim);
	}

	tensor tensor::broadcast_div(const tensor& a, const unsigned int dim) const
	{
		return operateBroadcast([](const tensor& a, const tensor& b) {
			return a / b;
			}, a, dim);
	}

	tensor tensor::dot(const tensor& a) const
	{
		//row를 단위로 병렬 처리, 3차원 이상인 경우 상위 2차원을 기준으로 행렬곱
		Shape inputShape = shape;
		Shape aShape = a.shape;
		//std::cout << "dot " << inputShape << ", " << aShape << std::endl;
		unsigned int m1_d = shape.dimension();
		unsigned int m2_d = a.shape.dimension();
		assert(m1_d > 1 && m2_d > 1 && m1_d == m2_d);
		assert(shape.dims[m1_d - 1] == a.shape.dims[m2_d - 2]);
		//m1[i, k] X m2[k, j] 에서의 i, j, k 
		unsigned int rows = shape.dims[m1_d - 2];
		unsigned int cols = a.shape.dims[m1_d - 1];
		unsigned int cPart = shape.dims[m1_d - 1];
		//std::cout << arraySize << ", " << cPart << std::endl;
		unsigned int totalRows = arraySize / cPart;
		std::shared_ptr<float[]> newArray = std::make_shared<float[]>(totalRows * cols);

		unsigned int cores = std::min(getNumCores(), totalRows);
		//std::cout << cores;
		std::vector<std::thread> threads;
		unsigned int blocks = totalRows / cores;
		unsigned int remainder = totalRows % cores;
		unsigned int start = 0;

		for (unsigned int t = 0; t < cores; ++t) {
			unsigned int end = start + blocks + (t < remainder ? 1 : 0);
			threads.emplace_back([start, end, rows, cols, cPart, m1_d, this, newArray, &a]() {
				for (unsigned int idx = start; idx < end; ++idx) {
					//현재 idx 이전의 행렬 수
					unsigned int batch = idx / rows;
					//현재 행
					unsigned int i = idx % rows;
					for (unsigned int j = 0; j < cols; ++j) {
						float sum = 0;
						for (unsigned int k = 0; k < cPart; ++k) {
							//오프셋 + 행렬 크기(행 x 열) * 현재 idx 이전의 행렬 개수 + 행 * 전체 열 + 열 로 인덱스 계산
							sum += array[offset + batch * rows * cPart + i * stride[m1_d - 2] + k * stride[m1_d - 1]] *
								a.array[a.offset + batch * cPart * cols + k * a.stride[m1_d - 2] + j * a.stride[m1_d - 1]];
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
		//std::cout << "new shape: " << newShape << std::endl;
		return tensor(newShape, newArray);
	}

	tensor tensor::transpose()
	{
		tensor result = *this;
		unsigned int size = shape.dimension();
		std::swap(result.shape.dims[size - 1], result.shape.dims[size - 2]);
		std::swap(result.stride[size - 1], result.stride[size - 2]);
		return result;
	}

	tensor tensor::slice(const unsigned int dim, const unsigned int start, const unsigned int end)
	{
		assert(start < end);
		assert(shape.dimension() > dim);
		Shape newShape = shape;
		newShape.dims[dim] = end - start;
		std::vector<int> startpos(shape.dimension(), 0);
		startpos[dim] = start;
		unsigned int newOffset = find_idx(startpos);
		tensor result = tensor(newShape, array);
		result.offset = newOffset;
		return result;
	}

	tensor tensor::sum(const unsigned int dim) const
	{
		assert(shape.dimension() > dim);
		Shape newShape = shape;
		newShape.dims[dim] = 1;

		unsigned int reduceSize = arraySize / shape.dims[dim];
		unsigned int cores = std::min(getNumCores(), reduceSize);
		unsigned int blockSize = reduceSize / cores;
		unsigned int remainder = reduceSize % cores;
		std::vector<std::thread> threads;
		std::shared_ptr<float[]> newArray = std::make_shared<float[]>(reduceSize);
		unsigned int start = 0;
		for (unsigned int i = 0; i < cores; i++)
		{
			unsigned int end = start + blockSize + (i < remainder ? 1 : 0);
			threads.emplace_back([start, end, dim, newArray, this] {
				std::vector<int> multi_idx(shape.dimension());
				for (unsigned int idx = start; idx < end; idx++)
				{
					get_multi_idx(multi_idx, idx);
					
					float sum = 0;
					for (unsigned int j = 0; j < shape.dims[dim]; j ++)
					{
						multi_idx[dim] = j;
						unsigned int flat_idx = find_idx(multi_idx);
						sum += array[flat_idx];
					}
					newArray[idx] = sum;
				}
				});
			start = end;
		}
		for (auto& t : threads) t.join();
		return tensor(newShape, newArray);
	}

	float tensor::sum() const
	{
		unsigned int cores = std::min(getNumCores(), (unsigned int)arraySize);
		unsigned int blockSize = arraySize / cores;
		unsigned int remainder = arraySize % cores;
		std::vector<std::thread> threads;
		std::atomic<float> result(0.0f);
		if (is_contiguous())
		{
			for (int i = 0; i < cores; i++)
			{
				unsigned int start = i * blockSize;
				unsigned int end = start + blockSize + (i < remainder ? 1 : 0);
				threads.emplace_back([start, end, &result, this] {
					for (unsigned int idx = start; idx < end; idx++)
					{
						result += array[offset + idx];
					}
					});
			}
		}
		else
		{
			for (int i = 0; i < cores; i++)
			{
				unsigned int start = i * blockSize;
				unsigned int end = start + blockSize + (i < remainder ? 1 : 0);
				threads.emplace_back([start, end, &result, this] {
					std::vector<int> multi_idx(shape.dimension(), 0);
					for (unsigned int idx = start; idx < end; idx++)
					{
						get_multi_idx(multi_idx, idx);
						result += array[find_idx(multi_idx)];
					}
					});
			}
		}
		for (auto& t : threads) t.join();
		return result.load();
	}

	float tensor::mean() const
	{
		return sum() / arraySize;
	}

	tensor tensor::concatenate(const unsigned int dim, const tensor& other)
	{
		return tensor();
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

	Shape tensor::getShape() const
	{
		return shape;
	}

	std::ostream& operator<<(std::ostream& out, tensor& t)
	{
		out << "tensor ";
		tensorPrint(0, 0, t.offset, t.shape.dims, t.array, out);
		return out;
	}

	void tensorPrint(unsigned int dim, unsigned int idx, unsigned int offset, std::vector<unsigned int>& dims, std::shared_ptr<float[]> array, std::ostream& out)
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
				tensorPrint(dim + 1, idx + blockSize*i, offset, dims, array, out);
				out << "]" ;
			}
		}
		else
		{
			//std::cout << "last dim" << std::endl;
			out << array[offset + idx];
			for (unsigned int i = 1; i < dims[dim]; i++)
			{
				//std::cout << idx + i;
				out << "," << array[offset + idx + i];
			}
		}
	}

	tensor exp(const tensor& a)
	{
		//std::cout << "exp" << std::endl;
		unsigned int cores = std::min(getNumCores(), (unsigned int)a.arraySize);
		std::shared_ptr<float[]> newArray = std::make_shared<float[]>(a.arraySize);
		unsigned int blockSize = a.arraySize / cores;
		unsigned int remainder = a.arraySize % cores;

		std::vector<std::thread> threads;

		unsigned int start = 0;

		for (unsigned int i = 0; i < cores; i++)
		{
			unsigned int end = start + blockSize + (i < remainder ? 1 : 0);
			threads.emplace_back([start, end, a, newArray] {
				for (int idx = start; idx < end; idx++)
				{
					//std::cout << a.array[idx] << std::endl;
					newArray[idx] = std::exp(a.array[idx]);
				}
				});
			start = end;
		}
		for (auto& t : threads) t.join();
		tensor t = tensor(a.shape, newArray);
		return t;
	}

	tensor max(const tensor& a, const tensor& b)
	{
		return a.operateTensor([](float a, float b) {
			return a > b ? a : b;
			}, b);
	}

	tensor min(const tensor& a, const tensor& b)
	{
		return a.operateTensor([](float a, float b) {
			return a < b ? a : b;
			}, b);
	}

	tensor max(const tensor& a, const float b)
	{
		return a.operateFloat([](float a, float b) {
			return a > b ? a : b;
			}, b);
	}

	tensor min(const tensor& a, const float b)
	{
		return a.operateFloat([](float a, float b) {
			return a < b ? a : b;
			}, b);
	}

	float uniformRandom(float min, float max)
	{
		static std::random_device rd;
		static std::mt19937 gen(rd());
		std::uniform_real_distribution<float> dis(min, max);
		return dis(gen);
	}

	tensor initUniformTensor(Shape shape, float min, float max)
	{
		int arraySize = 1;
		for (int i = 0; i < shape.dimension(); i++)
		{
			arraySize *= shape.dims[i];
		}
		std::shared_ptr<float[]> array = std::make_shared<float[]>(arraySize);
		for (int i = 0; i < arraySize; i++)
		{
			array[i] = uniformRandom(min, max);
		}
		return tensor(shape, array);
	}
}