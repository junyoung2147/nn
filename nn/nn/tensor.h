#pragma once
#include<vector>
#include <type_traits>
#include<iostream>
#include<memory>

namespace Tensor
{
	//텐서 모양 구조체
	struct Shape
	{
		std::vector<unsigned int> dims;
		int dimension() const { return dims.size(); }
		friend std::ostream& operator<<(std::ostream& out, Shape& s);
		bool operator==(const Shape& s) const;
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
		tensor operateFloat(float(*func)(float, float), float value) const;
		tensor operateTensor(float(*func)(float, float), const tensor& a) const;

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
		tensor slice(const unsigned int dim, const unsigned int start, const unsigned int end);
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
				std::cout << "return float " << find_idx(indices) << std::endl;
				return array[find_idx(indices)]; 
			}
			else {
				//텐서 반환
			}
		}
	};

	void tensorPrint(unsigned int dim, unsigned int idx, std::vector<unsigned int>& dims, std::shared_ptr<float[]> array, std::ostream& out);

	//현재 기기의 논리 코어 수를 반환
	unsigned int getNumCores();

	tensor exp(const tensor& a);

	tensor createFillTensor(Shape shape, float value);
	tensor createXiavierTensor(Shape shape, float gain = 1);
	tensor createNormalTensor(Shape shape, float mean = 0, float std = 1);
	tensor createHeTensor(Shape shape, float a = 0, float gain = 1);
}