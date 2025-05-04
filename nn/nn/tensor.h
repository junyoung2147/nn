#pragma once
#include<vector>
#include <type_traits>
#include<iostream>
#include<memory>

namespace Tensor
{
	//�ټ� ��� ����ü
	struct Shape
	{
		std::vector<unsigned int> dims;
		int dimension() const { return dims.size(); }
		friend std::ostream& operator<<(std::ostream& out, Shape& s);
		bool operator==(const Shape& s);
	};

	//�ټ� Ŭ����
	class tensor
	{
	private:
		Shape shape;
		std::shared_ptr<float[]> array;
		int arraySize;
		int find_idx(const std::vector<int>& indices);
		//��Ģ���� �ߺ� �ڵ� ���� ����
		tensor operateFloat(float(*func)(float, float), float value);
		tensor operateTensor(float(*func)(float, float), const tensor& a);

	public:
		tensor(Shape shape);
		tensor(Shape shape, std::shared_ptr<float[]> array);
		tensor(std::vector<unsigned int> v);
		/*tensor(const tensor& other);
		tensor(tensor&& other);
		~tensor();*/
		//tensor& operator=(tensor&& other) noexcept;
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
		decltype(auto) operator()(Indices... idxs) 
			//-> std::conditional_t<sizeof...(Indices) == this->shape.dimension(), float&, tensor&>
		{
			static_assert((std::is_integral_v<Indices> && ...), "�ε������� ������ ����");

			std::vector<int> indices = { static_cast<int>(idxs)... };

			if (sizeof...(Indices) == shape.dimension()) {
				std::cout << "return float " << find_idx(indices) << std::endl;
				return array[find_idx(indices)]; 
			}
			else {
				//�ټ� ��ȯ
			}
		}
	};

	void tensorPrint(unsigned int dim, unsigned int idx, std::vector<unsigned int>& dims, std::shared_ptr<float[]> array, std::ostream& out);

	//���� ����� �� �ھ� ���� ��ȯ
	unsigned int getNumCores();

	tensor createFillTensor(Shape shape, float value);
}