#pragma once
#include<iostream>
#include"tensor.h"

using namespace Tensor;

namespace nn
{
	class Layer
	{
	public:
		virtual ~Layer() = default;
		std::vector<tensor*> params;
		std::vector<tensor*> deltas;
		virtual tensor forward(const tensor& input) { return input; };
		virtual tensor backward(const tensor& grad_output) { return grad_output; };
		virtual void update(const float lr) {};
	};

	class Model
	{
	public:
		virtual tensor operator()(tensor input) = 0;
		virtual tensor backward(tensor grad_output) = 0;
	private:
	};

	class Sequential : public Model
	{
	public:
		std::vector<Layer*> layers;
		Sequential() {  };
		Sequential(std::vector<Layer*> layers) : layers(layers) {};
		void add(Layer* layer);
		tensor operator()(tensor input) override;
		tensor backward(tensor grad_output) override;
	};

	class Optimizer
	{
	public:
		virtual void step() = 0;
	private:
		//Optimizer();
	};

	class SGD : public Optimizer
	{
	private:
		std::vector<Layer*> layers;
		float lr;
	public:
		SGD(std::vector<Layer*> layers, float lr = 0.001) : layers(layers), lr(lr) {};
		virtual void step() override;
	};

	class Momentum : public Optimizer
	{
	private:
		std::vector<Layer*> layers;
		std::vector<std::vector<tensor>> velocity;
		float lr;
		float beta;
	public:
		Momentum(std::vector<Layer*> layers, float lr = 0.001, float beta = 0.9) : layers(layers), lr(lr), beta(beta)
		{
			velocity.reserve(layers.size());
			for (int i = 0; i < layers.size(); i++)
			{
				int layer_size = layers[i]->params.size();
				//std::cout << layer_size << std::endl;
				velocity.push_back(std::vector<tensor>(layer_size));
				for (int j = 0; j < layer_size; j++)
				{
					velocity[i][j] = tensor(layers[i]->params[j]->getShape());
				}
			}
		};
		virtual void step() override;
	};

	class Loss
	{
	public:
		virtual float operator()(const tensor& x, const tensor& y) = 0;
		virtual tensor backward(const tensor& x, const tensor& y) = 0;
	private:
		//Loss();
	};

	class MSE : public Loss
	{
	public:
		virtual float operator()(const tensor& x, const tensor& y) override;
		virtual tensor backward(const tensor& x, const tensor& y) override;
	};

	class CrossEntropy : public Loss
	{
	public:
		virtual float operator()(const tensor& x, const tensor& y) override;
		virtual tensor backward(const tensor& x, const tensor& y) override;
	private:
		tensor softmax(const tensor& t);
	};

	class Linear : public Layer
	{
	private:
		tensor w;
		tensor b;
		tensor delta_w;
		tensor delta_b;
		tensor input;
		tensor output;
	public:
		Linear(unsigned int in_dim, unsigned int out_dim, std::string init = "He");
		virtual tensor forward(const tensor& input) override;
		virtual tensor backward(const tensor& grad_output) override;
	};

	class ReLU : public Layer
	{
	private:
		tensor input;
	public:
		ReLU() = default;
		virtual tensor forward(const tensor& input) override;
		virtual tensor backward(const tensor& grad_output) override;
	};

	class Sigmoid : public Layer
	{
	private:
		tensor input;
		tensor sigmoid(const tensor& t);
	public:
		virtual tensor forward(const tensor& input) override;
		virtual tensor backward(const tensor& grad_output) override;
	};

	class Softmax : public Layer
	{
	private:
		tensor input;
	public:
		virtual tensor forward(const tensor& input) override;
		//virtual tensor backward(const tensor& grad_output) override;
	};
}