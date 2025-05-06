#pragma once
#include<iostream>
#include"tensor.h"

using namespace Tensor;

namespace nn
{
	class Sequential
	{

	};

	class Layer
	{
	public:
		virtual ~Layer() = default;
		virtual tensor forward(const tensor& input) { return input; };
		virtual tensor backward(const tensor& grad_output) { return grad_output; };
		virtual void update(const float lr) {};
	};

	class Linear : Layer
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
		virtual void update(const float lr) override;
	};

	class ReLU : Layer
	{
	private:
		tensor input;
	public:
		virtual tensor forward(const tensor& input) override;
		virtual tensor backward(const tensor& grad_output) override;
	};

	class Sigmoid : Layer
	{
	private:
		tensor input;
		tensor sigmoid(const tensor& t);
	public:
		virtual tensor forward(const tensor& input) override;
		virtual tensor backward(const tensor& grad_output) override;
	};
}