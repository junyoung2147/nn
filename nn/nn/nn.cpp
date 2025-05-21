#include"nn.h"
#include<assert.h>

namespace nn
{
	Linear::Linear(unsigned int in_dim, unsigned int out_dim, std::string init)
	{
		if (init == "Uniform")
		{
			w = initUniform(Shape({ in_dim, out_dim }), -0.1, 0.1);
			b = initUniform(Shape({ 1, out_dim }), -0.1, 0.1);
		}
		else if (init == "Normal")
		{
			w = initNormal(Shape({ in_dim, out_dim }), 0, 0.01);
			b = initNormal(Shape({ 1, out_dim }), 0, 0.01);
		}
		else if (init == "Xavier")
		{
			w = initXavier_normal(Shape({ in_dim, out_dim }), in_dim, out_dim);
			b = initXavier_normal(Shape({ 1, out_dim }), in_dim, out_dim);
		}
		else if (init == "Xavier_uniform")
		{
			w = initXavier_uniform(Shape({ in_dim, out_dim }), in_dim, out_dim);
			b = initXavier_uniform(Shape({ 1, out_dim }), in_dim, out_dim);
		}
		else if (init == "He_uniform")
		{
			w = initHe_uniform(Shape({ in_dim, out_dim }), in_dim);
			b = initHe_uniform(Shape({ 1, out_dim }), in_dim);
		}
		else if (init == "He")
		{
			w = initHe_normal(Shape({ in_dim, out_dim }), in_dim);
			b = initHe_normal(Shape({ 1, out_dim }), in_dim);
		}
		else
			assert(false && "Unknown initialization method");

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
		tensor exp_x = exp(input.broadcast_sub(input.max(1), 1));
		tensor sum_exp_x = exp_x.sum(1);
		return exp_x.broadcast_div(sum_exp_x + 1e-8f, 1);
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
				assert(param_shape == delta_shape);
				*layer->params[i] -= *(layer->deltas[i]) * lr;
				//std::cout << "After update: " << *(layer->params[i]) << std::endl;
			}
		}
	}

	void Momentum::step()
	{
		int i = 0;
		for (Layer* layer : layers)
		{
			for (int j = 0; j < layer->params.size(); j++)
			{
				//std::cout << j <<" " << i << std::endl;
				Shape param_shape = layer->params[j]->getShape();
				Shape delta_shape = layer->deltas[j]->getShape();
				Shape velocity_shape = velocity[i][j].getShape();
				//std::cout << "velocity shape: " << velocity_shape << std::endl;
				//std::cout << "param shape: " << param_shape << std::endl;
				assert(delta_shape == velocity_shape);
				assert(param_shape == delta_shape);
				//std::cout << "Before update: " << std::endl;
				velocity[i][j] = velocity[i][j] * beta + *(layer->deltas[j]) * lr;
				*layer->params[j] -= velocity[i][j];
			}
			i++;
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