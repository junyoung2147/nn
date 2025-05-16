#include<iostream>
#include"nn.h"

using namespace nn;

int main(void)
{
	Sequential model = Sequential();
	model.add(new Linear(10, 10));
	MSE mse = MSE();
	SGD sgd(model.layers);
	for (int i = 0; i < 10; i++)
	{
		tensor output = model(tensor());
		float loss = mse(output, tensor());
		tensor output_grad = mse.backward(output, tensor());
		model.backward(output_grad);
		sgd.step();
	}
}