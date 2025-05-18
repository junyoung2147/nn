#include"util.h"
#include<assert.h>

using namespace Tensor;

namespace nn
{
	DataLoader::DataLoader(tensor& x, tensor& y, unsigned int batch_size) : x(x), y(y)
	{
		assert(x.getShape().dims[0] == y.getShape().dims[0]);
		this->batch_size = batch_size;
		this->num_samples = x.getShape().dims[0];
		this->num_batchs = num_samples / batch_size;
	}

	std::pair<tensor, tensor> DataLoader::get()
	{
		unsigned int end = (index + batch_size) > num_samples ? num_samples : index + batch_size;
		tensor x_batch = x.slice(0, index, end);
		tensor y_batch = y.slice(0, index, end);
		index += batch_size;
		return std::pair<tensor, tensor>(x_batch, y_batch);
	}

	void DataLoader::reset()
	{
		index = 0;
	}

	bool DataLoader::has_next() const 
	{
		return index < num_samples;
	}

	unsigned int DataLoader::get_num_batchs() const
	{
		return num_batchs;
	}

	unsigned int DataLoader::get_num_samples() const
	{
		return num_samples;
	}

	unsigned int DataLoader::get_index() const
	{
		return index;
	}
};