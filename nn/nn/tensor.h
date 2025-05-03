#pragma once
#include<vector>
#include <type_traits>
#include<iostream>

namespace Tensor
{
	//텐서 모양 구조체
	struct Shape
	{
		std::vector<int> dims;
		int dimension() const { return dims.size(); }
		friend std::ostream& operator<<(std::ostream& out, Shape& s);
		bool operator==(const Shape& s);
	};

	//텐서 클래스
	class tensor
	{
	private:
		Shape shape;
		float* array;
		int arraySize;
		int find_idx(const std::vector<int>& indices);
		//사칙연산 중복 코드 따로 구현
		tensor operateFloat(float(*func)(float, float), float value);
		tensor operateTensor(float(*func)(float, float), const tensor& a);

	public:
		tensor(Shape shape);
		tensor(Shape shape, float* array) : shape(shape), array(array) {};
		tensor(std::vector<int> v);
		~tensor();
		void reshape(Shape& shape);
		Shape getShape();
		tensor dot(const tensor& B);
		tensor operator+(const float a);
		tensor operator-(const float a);
		tensor operator*(const float a);
		tensor operator/(const float a);
		tensor operator+(const tensor& a);
		tensor operator-(const tensor& a);
		tensor operator*(const tensor& a);
		tensor operator/(const tensor& a);
		friend std::ostream& operator<<(std::ostream& out, tensor& t);
		
		//가변 길이 템플릿으로 다차원 텐서의 인덱스 접근 구현, array(1,2,3) 형식
		template <typename... Indices>
		auto operator()(Indices... idxs) -> std::conditional_t<sizeof...(Indices) == this->shape.dimension(), float&, tensor&>
		{
			//static_assert((std::is_integral_v(Indices) && ...), "인덱스에는 정수만 가능");

			std::vector<int> indices = { static_cast<int>(idxs)... };

			if constexpr (sizeof...(Indices) == shape.dimension()) {
				return array[find_idx(indices)]; 
			}
			else {
				//텐서 반환
			}
		}
	};

	//현재 기기의 논리 코어 수를 반환
	unsigned int getNumCores();

	tensor createFillTensor(Shape shape, float value);
}