#pragma once
#include<vector>
#include <type_traits>
#include<iostream>
#include<memory>
#include<functional>
#include<thread>
#include<mutex>
#include<condition_variable>

namespace Tensor
{
	unsigned int getNumCores();

	//텐서 모양 구조체
	struct Shape
	{
		Shape(std::vector<unsigned int> dims) : dims(dims) {};
		Shape() {};
		std::vector<unsigned int> dims;
		int dimension() const { return dims.size(); }
		friend std::ostream& operator<<(std::ostream& out, Shape& s);
		bool operator==(const Shape& s) const;
	};

	struct ParallelJopInfo
	{
		std::shared_ptr<float[]> newArray;
		unsigned int numCores;
		unsigned int blockSize;
		unsigned int remainder;
	};

	class TensorThreadPool
	{
	public:
		static TensorThreadPool& getInstance()
		{
			//std::cout << getNumCores() << std::endl;
			static TensorThreadPool instance(getNumCores());
			return instance;
		}
		void run(int totalWork, std::function<void(int, int)> task);
		~TensorThreadPool();
	private:
		TensorThreadPool(int numThread);
		void workerThread(int threadId);
		std::vector<std::thread> threads;
		std::mutex mtx;
		std::condition_variable cv_start, cv_done;
		int numThreads;
		int totalWork;
		int activeThreads;
		std::function<void(int, int)> currentTask;
		bool stop_flag;
		bool run_flag;
		std::vector<bool> isActive;
	};

	//텐서 클래스
	class tensor
	{
	private:
		Shape shape;
		unsigned int offset;
		std::vector<unsigned int> stride;
		std::shared_ptr<float[]> array;
		int find_idx(const std::vector<int>& indices) const;
		void initStride();
		//사칙연산 중복 코드 따로 구현
		template<typename GetR>
		tensor applyBinaryContig(float(*func)(float, float), GetR get_r) const
		{
			//ParallelJopInfo info = prepareParallelJop(arraySize);
			std::shared_ptr<float[]> newArray = std::make_shared<float[]>(arraySize);
			TensorThreadPool::getInstance().run(arraySize, [newArray, func, get_r, this](int start, int end) {
						for (unsigned int idx = start; idx < end; idx++)
						{
							//인수로 받은 사칙연산 함수를 적용
							newArray[idx] = func(array[offset + idx], get_r(idx));
						}
						});
			return tensor(shape, newArray); //새로운 텐서 반환
		}
		template<typename GetR>
		tensor applyBinaryStrided(float(*func)(float, float), GetR get_r) const
		{
			//ParallelJopInfo info = prepareParallelJop(arraySize);
			std::shared_ptr<float[]> newArray = std::make_shared<float[]>(arraySize);
			TensorThreadPool::getInstance().run(arraySize, [get_r, newArray, func, this](int start, int end) {
				std::vector<int> multi_idx(shape.dimension(), 0);
				for (unsigned int idx = start; idx < end; idx++)
				{
					get_multi_idx(multi_idx, idx);
					newArray[idx] = func(array[find_idx(multi_idx)], get_r(multi_idx));
				}
				});
			return tensor(shape, newArray);
		}
		tensor applyBinaryFloat(float(*func)(float, float), float value) const;
		tensor applyBinaryTensor(float(*func)(float, float), const tensor& a) const;
		tensor applyUnary(float(*func)(float)) const;
		tensor applyReduction(std::function<void(std::shared_ptr<float[]>, std::vector<int>&, unsigned int idx)> func, const unsigned int dim) const;
		void updateFloat(float(*func)(float, float), float value) const;
		void updateTensor(float(*func)(float, float), const tensor& a) const;
		tensor applyBroadcast(tensor(*func)(const tensor&, const tensor&), const tensor& other, const unsigned int dim) const;
		bool is_contiguous() const;
		void get_multi_idx(std::vector<int>& multi_idx, unsigned int idx, int ignore_dim = -1) const;
		ParallelJopInfo prepareParallelJop(const unsigned int size) const;
	public:
		tensor();
		tensor(Shape shape);
		tensor(Shape shape, std::shared_ptr<float[]> array);
		tensor(std::vector<unsigned int> v);
		/*tensor(const tensor& other);
		tensor(tensor&& other);
		~tensor();*/
		//tensor& operator=(tensor&& other) noexcept;
		int arraySize;
		void reshape(Shape& shape);
		Shape getShape() const;
		tensor transpose();
		tensor dot(const tensor& a) const;
		tensor concatenate(const unsigned int dim, const tensor& other);
		tensor sum(const unsigned int dim) const;
		float sum() const;
		float mean() const;
		tensor max(const unsigned int dim) const;
		tensor argmax(const unsigned int dim) const;
		tensor slice(const unsigned int dim, const unsigned int start, const unsigned int end);
		tensor broadcast_add(const tensor& other, const unsigned int dim) const;
		tensor broadcast_sub(const tensor& other, const unsigned int dim) const;
		tensor broadcast_mul(const tensor& other, const unsigned int dim) const;
		tensor broadcast_div(const tensor& other, const unsigned int dim) const;
		tensor& operator+=(const float a);
		tensor& operator+=(const tensor& a);
		tensor& operator-=(const float a);
		tensor& operator-=(const tensor& a);
		tensor operator+(const float a) const;
		tensor operator-(const float a) const;
		friend tensor operator-(const float a, const tensor& b);
		tensor operator*(const float a) const;
		tensor operator/(const float a) const;
		friend tensor operator/(const float a, const tensor& b);
		tensor operator+(const tensor& a) const;
		tensor operator-(const tensor& a) const;
		tensor operator*(const tensor& a) const;
		tensor operator/(const tensor& a) const;
		tensor operator<(const tensor& other) const;
		tensor operator>(const tensor& other) const;
		tensor operator==(const tensor& other) const;
		tensor operator<(const float other) const;
		tensor operator>(const float other) const;
		tensor operator==(const float other) const;
		friend std::ostream& operator<<(std::ostream& out, tensor& t);
		friend tensor exp(const tensor& a);
		friend tensor log(const tensor& a);
		friend tensor max(const tensor& a, const tensor& b);
		friend tensor min(const tensor& a, const tensor& b);
		friend tensor max(const tensor& a, const float b);
		friend tensor min(const tensor& a, const float b);
		
		//가변 길이 템플릿으로 다차원 텐서의 인덱스 접근 구현, array(1,2,3) 형식
		template <typename... Indices>
		decltype(auto) operator()(Indices... idxs) const
			//-> std::conditional_t<sizeof...(Indices) == this->shape.dimension(), float&, tensor&>
		{
			static_assert((std::is_integral_v<Indices> && ...), "인덱스에는 정수만 가능");

			std::vector<int> indices = { static_cast<int>(idxs)... };

			if (sizeof...(Indices) == shape.dimension()) {
				//std::cout << "return float " << find_idx(indices) << std::endl;
				return array[find_idx(indices)]; 
			}
			else {
				//텐서 반환
			}
		}
	};

	void tensorPrint(unsigned int dim, unsigned int idx, unsigned int offset, std::vector<unsigned int>& dims, std::shared_ptr<float[]> array, std::ostream& out);

	//현재 기기의 논리 코어 수를 반환

	tensor exp(const tensor& a);

	float uniformRandom(float min, float max);
	float normalRandom(float mean, float std);

	tensor initFill(Shape shape, float value);
	tensor initUniform(Shape shape, float min = 0, float max = 1);
	tensor initXavier_normal(Shape shape, int in, int out, float gain = 1);
	tensor initXavier_uniform(Shape shape, int in, int out, float gain = 1);
	tensor initNormal(Shape shape, float mean = 0, float std = 1);
	tensor initHe_normal(Shape shape, int in, float gain = 1);
	tensor initHe_uniform(Shape shape, int in, float gain = 1);
}