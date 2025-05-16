#include"nn.h"

namespace nn
{
	Linear::Linear(unsigned int in_dim, unsigned int out_dim)
	{
		w = tensor({in_dim, out_dim});
		b = tensor({ 1, out_dim });
		params.push_back(w);
		params.push_back(b);
		delta_w = tensor();
		delta_b = tensor();
		deltas.push_back(delta_w);
		deltas.push_back(delta_b);
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

	void Sequential::add(Layer* layer)
	{
		layers.push_back(layer);
	}

	tensor Sequential::operator()(tensor input)
	{
		tensor output = input;
		for (Layer* layer : layers)
		{
			output = layer->forward(output);
		}
		return output;
	}

	tensor Sequential::backward(tensor grad_output)
	{
		tensor output = grad_output;
		for (auto it = layers.rbegin(); it != layers.rend(); ++it)
		{
			output = (*it)->backward(output);
		}
		return output;
	}

	void SGD::step()
	{
		for (Layer* layer : layers)
		{
			for (int i = 0; i < layer->params.size(); i++)
			{
				layer->params[i] = layer->params[i] - layer->deltas[i] * lr;
			}
		}
	}

	float MSE::operator()(const tensor& x, const tensor& y)
	{
		return 0;
	}

	tensor MSE::backward(const tensor& x, const tensor& y)
	{
		return (x - y) * (2 / (float)x.arraySize);
	}
}