#include<iostream>
#include<fstream>
#include"nn.h"
#include"util.h"
#include<memory>

using namespace nn;

tensor load_mnist_data(const std::string& image_path, unsigned int num_samples) {
	const unsigned int IMAGE_SIZE = 28 * 28;
	const unsigned int HEADER_SIZE = 16;
	const unsigned int data_size = num_samples * IMAGE_SIZE;
	std::ifstream data_file(image_path, std::ios::binary);
	if (!data_file.is_open()) {
		std::cerr << "Error opening files!" << std::endl;
		return tensor();
	}
	unsigned char* data_buffer = new unsigned char[num_samples * IMAGE_SIZE];
	data_file.seekg(16); // Skip the header
	data_file.read((char*)data_buffer, data_size);
	data_file.close();

	std::shared_ptr<float[]> data = std::make_shared<float[]>(data_size);
	for (int i = 0; i < data_size; i++)
	{
		data[i] = (float)data_buffer[i] / 255.0f;
	}
	tensor input_data = tensor(Shape({ num_samples, IMAGE_SIZE }), data);

	return input_data;
}

tensor load_mnist_labels(const std::string& label_path, unsigned int num_samples) {
	const unsigned int HEADER_SIZE = 8;
	std::ifstream label_file(label_path, std::ios::binary);
	if (!label_file.is_open()) {
		std::cerr << "Error opening files!" << std::endl;
		return tensor();
	}
	unsigned char* label_buffer = new unsigned char[num_samples];
	label_file.seekg(8); // Skip the header
	label_file.read((char*)label_buffer, num_samples);
	label_file.close();
	std::shared_ptr<float[]> labels = std::make_shared<float[]>(num_samples * 10);
	for (int i = 0; i < num_samples; i++)
	{
		labels[i * 10 + label_buffer[i]] = 1;
	}
	tensor label_data = tensor(Shape({ num_samples, 10}), labels);
	return label_data;
}

int main(void)
{
	Sequential model = Sequential();
	std::cout << "Sequential model created" << std::endl;
	model.add(new Linear(28*28, 128));
	model.add(new ReLU());
	model.add(new Linear(128, 64));
	model.add(new ReLU());
	model.add(new Linear(64, 64));
	model.add(new ReLU());
	model.add(new Linear(64, 10));
	//std::cout << *model.layers[0]->params[0] << std::endl;

	std::cout << "Layers added to model" << std::endl;
	MSE mse = MSE();
	SGD sgd(model.layers,0.0005);
	std::cout << "SGD optimizer created" << std::endl;
	
	tensor input = load_mnist_data("C:\\Users\\user\\Desktop\\train-images-idx3-ubyte\\train-images-idx3-ubyte", 8000);
	tensor label = load_mnist_labels("C:\\Users\\user\\Desktop\\train-labels-idx1-ubyte\\train-labels-idx1-ubyte", 8000);
	tensor input_1 = label.slice(0, 0, 1);
	std::cout << input_1 << std::endl;
	tensor input_2 = label.slice(0, 10, 11);
	std::cout << input_2 << std::endl;
	
	//DataLoader data_loader = DataLoader(input, label, 64);
	//for (int i = 0; i < 10; i++)
	//{
	//	std::cout << "Epoch: " << i << std::endl;
	//	while (data_loader.has_next())
	//	{
	//		std::pair<tensor, tensor> data = data_loader.get();
	//		//std::cout << "Data: " << data.first << std::endl;
	//		tensor output = model(data.first);
	//		//std::cout << "Output: " << output << std::endl;
	//		//float loss = mse(output, data.second);
	//		tensor output_grad = mse.backward(output, data.second);
	//		model.backward(output_grad);
	//		sgd.step();
	//		//std::cout << "Loss: " << mse(output, data.second) << std::endl;
	//	}
	//	data_loader.reset();
	//}

	//tensor test_out = model(input_1);
	//std::cout << "Test output: " << test_out << std::endl;
}