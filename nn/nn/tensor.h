#pragma once
#include<vector>
#include <type_traits>
#include<iostream>

namespace Tensor
{
	//�ټ� ��� ����ü
	struct Shape
	{
		std::vector<int> dims;
		int dimension() const { return dims.size(); }
		friend std::ostream& operator<<(std::ostream& out, Shape& s);
		bool operator==(const Shape& s);
	};

	//�ټ� Ŭ����
	class tensor
	{
	private:
		Shape shape;
		float* array;
		int arraySize;
		int find_idx(const std::vector<int>& indices);
		//��Ģ���� �ߺ� �ڵ� ���� ����
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
		
		//���� ���� ���ø����� ������ �ټ��� �ε��� ���� ����, array(1,2,3) ����
		template <typename... Indices>
		auto operator()(Indices... idxs) -> std::conditional_t<sizeof...(Indices) == this->shape.dimension(), float&, tensor&>
		{
			//static_assert((std::is_integral_v(Indices) && ...), "�ε������� ������ ����");

			std::vector<int> indices = { static_cast<int>(idxs)... };

			if constexpr (sizeof...(Indices) == shape.dimension()) {
				return array[find_idx(indices)]; 
			}
			else {
				//�ټ� ��ȯ
			}
		}
	};

	//���� ����� �� �ھ� ���� ��ȯ
	unsigned int getNumCores();

	tensor createFillTensor(Shape shape, float value);
}