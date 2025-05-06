#include"nn.h"

namespace nn
{
	Linear::Linear(unsigned int in_dim, unsigned int out_dim)
	{
		w = tensor({in_dim, out_dim});
		b = tensor({ 1, out_dim });
		delta_w = tensor();
		delta_b = tensor();
		input = tensor();
		output = tensor();
	}

	tensor Linear::forward(const tensor& input)
	{
		this->input = input;
		output = input.dot(w) + b;
		return output;
	}

	tensor Linear::backward(const tensor& grad_output)
	{
		delta_b = grad_output;
		delta_w = input.transpose().dot(grad_output);
		return grad_output.dot(w.transpose());
	}

	void Linear::update(const float lr)
	{
		w = w - delta_w * lr;
		b = b - delta_b * lr;
		delta_w = tensor();
		delta_b = tensor();
	}

	tensor ReLU::forward(const tensor& input)
	{
		this->input = input;
		return max(input, 0);
	}

	tensor ReLU::backward(const tensor& grad_output)
	{
		return grad_output * (input > 0);
	}

	tensor Sigmoid::sigmoid(const tensor& t)
	{
		return 1 / (exp(input) + 1);
	}

	tensor Sigmoid::forward(const tensor& input)
	{
		this->input = input;
		return sigmoid(input);
	}

	tensor Sigmoid::backward(const tensor& grad_output)
	{
		tensor s = sigmoid(input);
		return grad_output * s * (1 - s);
	}
}