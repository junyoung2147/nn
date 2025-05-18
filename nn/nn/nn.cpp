#include"nn.h"
#include<assert.h>

namespace nn
{
	Linear::Linear(unsigned int in_dim, unsigned int out_dim)
	{
		w = initUniformTensor(Shape({in_dim, out_dim}), -0.1, 0.1);
		b = initUniformTensor(Shape({ 1, out_dim }), -0.1, 0.1);
		Shape s = w.getShape();
		std::cout << s << std::endl;
		params.push_back(&w);
		params.push_back(&b);
		delta_w = tensor();
		delta_b = tensor();
		deltas.push_back(&delta_w);
		deltas.push_back(&delta_b);
		input = tensor();
		output = tensor();
	}

	tensor Linear::forward(const tensor& input)
	{
		this->input = input;
		output = input.dot(w).broadcast_add(b, 0);
		//std::cout << "output: " << output << std::endl;
		return output; // [batch_size, out_dim]
	}

	tensor Linear::backward(const tensor& grad_output)
	{
		delta_b = grad_output.sum(0);
		//std::cout << delta_b << input << std::endl;
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
		/*tensor temp = t;
		std::cout << temp << std::endl;*/
		return 1 / (exp(t * -1) + 1);
	}

	tensor Sigmoid::forward(const tensor& input)
	{
		this->input = input;
		return sigmoid(input);
	}

	tensor Sigmoid::backward(const tensor& grad_output)
	{
		tensor s = sigmoid(input);
		/*tensor temp = grad_output * s * (1 - s);
		std::cout << "sigmoid backward " << s << std::endl;*/
		return grad_output * s * (1 - s);
	}

	tensor Softmax::forward(const tensor& input)
	{
		this->input = input;
		tensor exp_x = exp(input);
		tensor sum_exp_x = exp_x.sum(0);
		return exp_x.broadcast_div(sum_exp_x, 1);
	}

	void Sequential::add(Layer* layer)
	{
		std::cout << "Adding layer: " << std::endl;
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
				Shape param_shape = layer->params[i]->getShape();
				Shape delta_shape = layer->deltas[i]->getShape();
				*layer->params[i] -= *(layer->deltas[i]) * lr;
				//std::cout << "After update: " << *(layer->params[i]) << std::endl;
			}
		}
	}

	float MSE::operator()(const tensor& x, const tensor& y)
	{
		assert(x.getShape() == y.getShape());
		Shape x_shape = x.getShape();
		//std::cout << x_shape << std::endl;
		tensor diff = x - y;
		//std::cout << x.arraySize << std::endl;
		return (diff * diff).sum() / (float)y.arraySize;
	}

	tensor MSE::backward(const tensor& x, const tensor& y)
	{
		assert(x.getShape() == y.getShape());
		/*tensor temp = (x - y) * (2 / (float)x.arraySize);
		std::cout << "backward loss " << temp << std::endl;*/
		return (x - y) * (2 / (float)x.arraySize);
	}

	float CrossEntropy::operator()(const tensor& x, const tensor& y)
	{
		assert(x.getShape() == y.getShape());
		tensor softmax_x = softmax(x);
		//std::cout << "softmax_x: " << softmax_x << std::endl;
		return (y * log(softmax(x) + 1e-8f) * -1).mean();
	}

	tensor CrossEntropy::backward(const tensor& x, const tensor& y)
	{
		assert(x.getShape() == y.getShape());
		tensor pridict = softmax(x);
		//std::cout << "pridict: " << pridict << std::endl;
		return (pridict - y) / x.getShape().dims[0];
	}

	tensor CrossEntropy::softmax(const tensor& x)
	{
		//tensor max_x = x.max(1);
		//std::cout << "sum_x: " << max_x << std::endl;
		tensor exp_x = exp(x.broadcast_sub(x.max(1), 1));
		tensor sum_exp_x = exp_x.sum(1);
		//tensor sum_x = exp_x.sum(1);
		//std::cout << "sum_exp_x: " << sum_x << std::endl;
		return exp_x.broadcast_div(sum_exp_x + 1e-8f, 1);
	}
}