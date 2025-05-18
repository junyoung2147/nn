#pragma once
#include"tensor.h"

using namespace Tensor;

namespace nn
{
	class DataLoader
	{
	private:
		tensor& x;
		tensor& y;
		unsigned int batch_size;
		unsigned int index = 0;
		unsigned int num_samples;
		bool shuffle = false;
		unsigned int num_batchs;
	public:
		DataLoader(tensor& x, tensor& y, unsigned int batch_size);
		std::pair<tensor, tensor> get();
		void reset();
		bool has_next() const;
		unsigned int get_num_batchs() const;
		unsigned int get_index() const;
		unsigned int get_num_samples() const;
	};
};