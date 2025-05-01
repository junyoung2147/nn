#pragma once

namespace nn
{
	class Layer
	{
	public:
		virtual ~Layer();
		virtual void forward();
		virtual void backward();
	};
}