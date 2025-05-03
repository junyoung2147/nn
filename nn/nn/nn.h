#pragma once

namespace nn
{
	class Sequential
	{

	};

	class Layer
	{
	public:
		virtual ~Layer();
		virtual void forward();
		virtual void backward();
	};

	class Linear : Layer
	{
	public:
		Linear(int in_dim, int out_dim);
	};


}