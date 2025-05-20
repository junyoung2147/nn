#include "tensor.h"
#include<assert.h>
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

	void tensor::get_multi_idx(std::vector<int>& multi_idx, unsigned int idx, int ignore_dim) const
	{
		for (int i = shape.dimension() - 1; i >= 0; i--)
		{
			if (i == ignore_dim)
			{
				multi_idx[i] = 0;
				continue;
			}
			multi_idx[i] = idx % shape.dims[i];
			idx /= shape.dims[i];
		}
	}

	ParallelJopInfo tensor::prepareParallelJop(const unsigned int size) const
	{
		ParallelJopInfo info;
		info.newArray = std::make_shared<float[]>(size);
		info.numCores = getNumCores();
		info.blockSize = size / info.numCores;
		info.remainder = size % info.numCores;
		return info;
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

	tensor tensor::applyBinaryFloat(float(*func)(float, float), float value) const
	{
		if (is_contiguous())
		{
			return applyBinaryContig(func, [value](unsigned int) {return value; });
		}
		else
		{
			return applyBinaryStrided(func, [value](std::vector<int>) {return value; });
		}
	}

	tensor tensor::operator+(const float a) const
	{
		return applyBinaryFloat([](float a, float b) {return a + b; }, a);
	}

	tensor tensor::operator-(const float a) const
	{
		return applyBinaryFloat([](float a, float b) {return a - b; }, a);
	}

	tensor operator-(const float a, const tensor& b)
	{
		return b.applyBinaryFloat([](float a, float b) {return b - a; }, a);
	}

	tensor tensor::operator*(const float a) const
	{
		return applyBinaryFloat([](float a, float b) {return a * b; }, a);
	}

	tensor tensor::operator/(const float a) const
	{
		return applyBinaryFloat([](float a, float b) {return a / b; }, a);
	}

	tensor operator/(const float a, const tensor& b)
	{
		return b.applyBinaryFloat([](float a, float b) {return b / a; }, a);
	}

	tensor tensor::applyBinaryTensor(float(*func)(float, float), const tensor& a) const
	{
		assert(this->shape == a.shape);
	
		if (is_contiguous() && a.is_contiguous())
		{
			return applyBinaryContig(func, [a](unsigned int idx) {return a.array[a.offset + idx]; });
		}
		else
		{
			return applyBinaryStrided(func, [a](std::vector<int> multi_idx) {return a.array[a.find_idx(multi_idx)]; });
		}
	}

	tensor tensor::operator+(const tensor& a) const
	{
		return applyBinaryTensor([](float a, float b) {return a + b; }, a);
	}

	tensor tensor::operator-(const tensor& a) const
	{
		return applyBinaryTensor([](float a, float b) {return a - b; }, a);
	}

	tensor tensor::operator*(const tensor& a) const
	{
		return applyBinaryTensor([](float a, float b) {return a * b; }, a);
	}

	tensor tensor::operator/(const tensor& a) const
	{
		return applyBinaryTensor([](float a, float b) {return a / b; }, a);
	}

	tensor tensor::operator<(const tensor& other) const
	{
		return applyBinaryTensor([](float a, float b)->float {return a < b; }, other);
	}

	tensor tensor::operator>(const tensor& other) const
	{
		return applyBinaryTensor([](float a, float b)->float {return a > b; }, other);
	}

	tensor tensor::operator==(const tensor& other) const
	{
		return applyBinaryTensor([](float a, float b)->float {return a == b; }, other);
	}

	tensor tensor::operator<(const float other) const
	{
		return applyBinaryFloat([](float a, float b)->float {return a < b; }, other);
	}

	tensor tensor::operator>(const float other) const
	{
		return applyBinaryFloat([](float a, float b)->float {return a > b; }, other);
	}

	tensor tensor::operator==(const float other) const
	{
		return applyBinaryFloat([](float a, float b)->float {return a == b; }, other);
	}

	void tensor::updateTensor(float(*func)(float, float), const tensor& a) const
	{
		assert(this->shape == a.shape);
		
		if (is_contiguous() && a.is_contiguous())
		{
			TensorThreadPool::getInstance().run(arraySize, [a, func, this](int start, int end) {
				for (unsigned int idx = start; idx < end; idx++)
				{
					this->array[offset + idx] = func(array[offset + idx], a.array[a.offset + idx]);
				}
				});
		}
		else
		{
			TensorThreadPool::getInstance().run(arraySize, [a, func, this](int start, int end) {
				std::vector<int> multi_idx(shape.dimension(), 0);
				for (unsigned int idx = start; idx < end; idx++)
				{
					get_multi_idx(multi_idx, idx);
					this->array[offset + idx] = func(array[find_idx(multi_idx)], a.array[a.find_idx(multi_idx)]);
				}
				});
		}
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
		if (is_contiguous())
		{
			TensorThreadPool::getInstance().run(arraySize, [a, func, this](int start, int end) {
				for (unsigned int idx = start; idx < end; idx++)
				{
					this->array[offset + idx] = func(array[offset + idx], a);
				}
				});
		}
		else
		{
			TensorThreadPool::getInstance().run(arraySize, [a, func, this](int start, int end) {
				std::vector<int> multi_idx(shape.dimension(), 0);
				for (unsigned int idx = start; idx < end; idx++)
				{
					get_multi_idx(multi_idx, idx);
					this->array[offset + idx] = func(array[find_idx(multi_idx)], a);
				}
				});
		}
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

	tensor tensor::applyBroadcast(tensor(*func)(const tensor&, const tensor&), const tensor& a, const unsigned int dim) const
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
		return applyBroadcast([](const tensor& a, const tensor& b) {
			return a + b;
			}, a, dim);
	}

	tensor tensor::broadcast_sub(const tensor& a, const unsigned int dim) const
	{
		return applyBroadcast([](const tensor& a, const tensor& b) {
			return a - b;
			}, a, dim);
	}

	tensor tensor::broadcast_mul(const tensor& a, const unsigned int dim) const
	{
		return applyBroadcast([](const tensor& a, const tensor& b) {
			return a * b;
			}, a, dim);
	}

	tensor tensor::broadcast_div(const tensor& a, const unsigned int dim) const
	{
		return applyBroadcast([](const tensor& a, const tensor& b) {
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

		TensorThreadPool::getInstance().run(totalRows, [rows, cols, cPart, m1_d, this, newArray, &a](int start, int end) {
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

	tensor tensor::applyReduction(std::function<void(std::shared_ptr<float[]>, std::vector<int>&, unsigned int idx)> func, const unsigned int dim) const
	{
		assert(shape.dimension() > dim);
		Shape newShape = shape;
		newShape.dims[dim] = 1;

		unsigned int reduceSize = arraySize / shape.dims[dim];
		std::shared_ptr<float[]> newArray = std::make_shared<float[]>(reduceSize);

		TensorThreadPool::getInstance().run(reduceSize, [dim, this, func, newArray](int start, int end) {
			std::vector<int> multi_idx(shape.dimension());
			for (unsigned int idx = start; idx < end; idx++)
			{
				get_multi_idx(multi_idx, idx, dim);
				func(newArray, multi_idx, idx);
			}
			});
		return tensor(newShape, newArray);
	}

	tensor tensor::sum(const unsigned int dim) const
	{
		return applyReduction([this, dim](std::shared_ptr<float[]> newArray, std::vector<int>& multi_idx, unsigned int idx) {
			float sum = 0;
			for (unsigned int j = 0; j < shape.dims[dim]; j++)
			{
				multi_idx[dim] = j;
				unsigned int flat_idx = find_idx(multi_idx);
				sum += array[flat_idx];
			}
			newArray[idx] = sum;
			}, dim);
	}

	float tensor::sum() const
	{
		std::atomic<float> result(0.0f);
		if (is_contiguous())
		{
			TensorThreadPool::getInstance().run(arraySize, [&result, this](int start, int end) {
				for (unsigned int idx = start; idx < end; idx++)
				{
					result += array[offset + idx];
				}
				});
		}
		else
		{
			TensorThreadPool::getInstance().run(arraySize, [&result, this](int start, int end) {
				std::vector<int> multi_idx(shape.dimension(), 0);
				for (unsigned int idx = start; idx < end; idx++)
				{
					get_multi_idx(multi_idx, idx);
					result += array[find_idx(multi_idx)];
				}
				});
		}
		return result.load();
	}

	tensor tensor::max(const unsigned int dim) const
	{
		return applyReduction([this, dim](std::shared_ptr<float[]> newArray, std::vector<int>& multi_idx, unsigned int idx) {
			float maxValue = array[find_idx(multi_idx)];
			for (unsigned int j = 0; j < shape.dims[dim]; j++)
			{
				multi_idx[dim] = j;
				unsigned int flat_idx = find_idx(multi_idx);
				maxValue = std::max(maxValue, array[flat_idx]);
			}
			newArray[idx] = maxValue;
			}, dim);
	}

	tensor tensor::argmax(const unsigned int dim) const
	{
		return applyReduction([this, dim](std::shared_ptr<float[]> newArray, std::vector<int>& multi_idx, unsigned int idx) {
			float maxValue = array[find_idx(multi_idx)];
			unsigned int maxIdx = 0;
			for (unsigned int j = 0; j < shape.dims[dim]; j++)
			{
				multi_idx[dim] = j;
				unsigned int flat_idx = find_idx(multi_idx);
				if (maxValue < array[flat_idx])
				{
					maxIdx = j;
					maxValue = array[flat_idx];
				}
			}
			//std::cout << maxValue << std::endl;
			newArray[idx] = (float)maxIdx;
			},dim);
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

	tensor tensor::applyUnary(float(*func)(float)) const
	{
		std::shared_ptr<float[]> newArray = std::make_shared<float[]>(arraySize);
		if (is_contiguous())
		{
			TensorThreadPool::getInstance().run(arraySize, [this, func, newArray](int start, int end) {
				for (unsigned int idx = start; idx < end; idx++)
				{
					newArray[idx] = func(array[offset + idx]);
				}
				});
		}
		else
		{
			TensorThreadPool::getInstance().run(arraySize, [this, func, newArray](int start, int end) {
				std::vector<int> multi_idx(shape.dimension(), 0);
				for (unsigned int idx = start; idx < end; idx++)
				{
					get_multi_idx(multi_idx, idx);
					newArray[idx] = func(array[find_idx(multi_idx)]);
				}
				});
		}
		return tensor(shape, newArray);
	}

	tensor exp(const tensor& a)
	{
		//std::cout << "exp" << std::endl;
		return a.applyUnary([](float a) {
			return std::exp(a);
			});
	}

	tensor log(const tensor& a)
	{
		return a.applyUnary([](float a) {
			return std::log(a);
			});
	}

	tensor max(const tensor& a, const tensor& b)
	{
		return a.applyBinaryTensor([](float a, float b) {
			return a > b ? a : b;
			}, b);
	}

	tensor min(const tensor& a, const tensor& b)
	{
		return a.applyBinaryTensor([](float a, float b) {
			return a < b ? a : b;
			}, b);
	}

	tensor max(const tensor& a, const float b)
	{
		return a.applyBinaryFloat([](float a, float b) {
			return a > b ? a : b;
			}, b);
	}

	tensor min(const tensor& a, const float b)
	{
		return a.applyBinaryFloat([](float a, float b) {
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

	void TensorThreadPool::run(int totalWork, std::function<void(int, int)> task)
	{
		{
			std::unique_lock<std::mutex> lock(mtx);
			this->totalWork = totalWork;
			this->currentTask = task;
			this->activeThreads = numThreads;
			this->run_flag = true;
			for (int i = 0; i < numThreads; ++i)
			{
				isActive[i] = true;
			}
		}
		cv_start.notify_all();

		std::unique_lock<std::mutex> lock(mtx);
		cv_done.wait(lock, [this] { return activeThreads == 0; });
		//std::cout << "all done" << std::endl;
	}

	void TensorThreadPool::workerThread(int threadId)
	{
		/*std::unique_lock<std::mutex> lock(mtx);
		std::cout << threadId << " thread start" << std::endl;
		lock.unlock();*/
		while (true)
		{
			std::unique_lock<std::mutex> lock(mtx);
			cv_start.wait(lock, [this, threadId] { return stop_flag || (run_flag && isActive[threadId]); });
			if (stop_flag) return;
			int blockSize = (totalWork + numThreads - 1) / numThreads;
			int start = threadId * blockSize;
			int end = std::min(start + blockSize, totalWork);
			lock.unlock();

			if (start < totalWork)
			{
				currentTask(start, end);
			}
			lock.lock();
			isActive[threadId] = false;
			if (--activeThreads == 0)
			{
				run_flag = false;
				cv_done.notify_one();
			}
		}
	}

	TensorThreadPool::TensorThreadPool(int numThreads) : numThreads(numThreads), run_flag(false), stop_flag(false), activeThreads(0), totalWork(0)
	{
		isActive = std::vector(numThreads, false);
		for (int i = 0; i < numThreads; ++i)
		{
			threads.emplace_back(&TensorThreadPool::workerThread, this, i);
		}
	}

	TensorThreadPool::~TensorThreadPool()
	{
		stop_flag = true;
		run_flag = true;
		cv_start.notify_all();
		for (auto& thread : threads)
		{
			if (thread.joinable()) thread.join();
		}
	}
}