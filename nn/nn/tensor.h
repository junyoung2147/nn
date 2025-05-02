#pragma once
#include<vector>
#include <type_traits>
#include<iostream>

namespace Tensor
{
	struct Shape
	{
		const int dimension;
		const int* shape;
		friend std::ostream& operator<<(std::ostream& out, Shape& s);
		bool operator==(Shape& s);
	};

	class tensor
	{
	private:
		Shape& shape;
		float* array;
		int arraySize;
		int find_idx(const std::vector<int>& indices);
		tensor& operateFloat(float(*func)(float, float), float value);
		tensor& operateTensor(float(*func)(float, float), const tensor& a);

	public:
		tensor(Shape& shape);
		tensor(Shape& shape, float* array) : shape(shape), array(array) {};
		tensor& dot(tensor& B);
		tensor& operator+(const float a);
		tensor& operator-(const float a);
		tensor& operator*(const float a);
		tensor& operator/(const float a);
		tensor& operator+(const tensor& a);
		tensor& operator-(const tensor& a);
		tensor& operator*(const tensor& a);
		tensor& operator/(const tensor& a);
		friend std::ostream& operator<<(std::ostream& out, tensor& t);
		
		template <typename... Indices>
		auto operator()(Indices... idxs) -> std::conditional_t<sizeof...(Indices) == this->shape.dimension, float&, tensor&>
		{
			static_assert((std::is_integral_v(Indices) && ...), "인덱스에는 정수만 가능");

			std::vector<int> indices = { static_cast<int>(idxs)... };

			if constexpr (sizeof...(Indices) == shape.size()) {
				return array[find_idx(indices)]; 
			}
			else {
				//텐서 반환
			}
		}
	};

	unsigned int getNumCores();

	tensor& createFillTensor(Shape& shape, float value);
}