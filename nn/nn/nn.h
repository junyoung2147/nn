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

	class Sequential
	{
	public:
		std::vector<Layer*> layers;
		Sequential() {  };
		Sequential(std::vector<Layer*> layers) : layers(layers) {};
		void add(Layer* layer);
		tensor operator()(tensor input);
		tensor backward(tensor grad_output);
	};

	class Model
	{
	public:
		virtual void forward(tensor input) {};
		virtual void backword(tensor grad_output) {};
	private:
		Model();
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
		Linear(unsigned int in_dim, unsigned int out_dim);
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
		virtual tensor backward(const tensor& grad_output) override;
	};
}