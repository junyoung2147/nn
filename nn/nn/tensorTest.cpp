#include<iostream>
#include"tensor.h"

using namespace Tensor;

int main(void)
{
	tensor t({ 2,2,2 });
	std::cout << t;
	t(1,1,1) = 20;
	std::cout << t(1, 1, 1) << std::endl;
	std::cout << t;
	tensor t1 = t * 10;
	std::cout << t1;
	t1 = t1.dot(t);
	std::cout << t1;
}