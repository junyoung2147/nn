#include<iostream>
#include<fstream>
#include"nn.h"
#include"util.h"
#include<memory>
#include<ctime>

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

Sequential build_model(std::string init) {
	Sequential model = Sequential();
	model.add(new Linear(28 * 28, 128, init));
	model.add(new ReLU());
	model.add(new Linear(128, 64, init));
	model.add(new ReLU());
	model.add(new Linear(64, 10, init));
	return model;
}

void mnistTest(void)
{
	Sequential model = build_model("He");
	std::cout << "Sequential model created" << std::endl;
	MSE mse = MSE();
	CrossEntropy cross_entropy = CrossEntropy();
	//SGD sgd(model.layers,0.01);
	Momentum momentum(model.layers, 0.01, 0.9);
	std::cout << "SGD optimizer created" << std::endl;
	
	tensor input = load_mnist_data("C:\\Users\\user\\Desktop\\train-images-idx3-ubyte\\train-images-idx3-ubyte", 1000);
	tensor label = load_mnist_labels("C:\\Users\\user\\Desktop\\train-labels-idx1-ubyte\\train-labels-idx1-ubyte", 1000);
	
	int batch_size = 32;
	DataLoader data_loader = DataLoader(input, label, batch_size);
	
	float sum_loss = 0;
	int num_correct = 0;
	time_t start_time = 0;
	for (int i = 0; i < 30; i++)
	{
		start_time = clock();
		while (data_loader.has_next())
		{
			std::pair<tensor, tensor> data = data_loader.get();
			//std::cout << "Data: " << data.first << std::endl;
			tensor output = model(data.first);
			float loss = cross_entropy(output, data.second);
			sum_loss += loss;
			tensor output_grad = cross_entropy.backward(output, data.second);
			model.backward(output_grad);
			//sgd.step();
			momentum.step();
			std::cout << "batch " << data_loader.get_index() / batch_size << "/" << data_loader.get_num_batchs() << " Loss: " << loss << std::endl;
			tensor pred = output.argmax(1);
			tensor label = data.second.argmax(1);
			//std::cout << output << std::endl;
			/*std::cout << "Pred: " << pred << std::endl;
			std::cout << "Label: " << label << std::endl;*/
			num_correct += (pred == label).sum();
		}
		data_loader.reset();
		std::cout << "Epoch: " << i << " - " << (clock() - start_time) / 1000 << "s " << " - average loss: " << sum_loss / data_loader.get_num_batchs();
		std::cout << " - accuracy: " << (float)num_correct / data_loader.get_num_samples() << std::endl;
		num_correct = 0;
		sum_loss = 0;
	}

	DataLoader test_data_loader = DataLoader(input, label, 1);
	model.add(new Softmax());
	for (int i = 0; i < 10; i++)
	{
		std::pair<tensor, tensor> data = test_data_loader.get();
		tensor output = model(data.first);
		int pred = output.argmax(1)(0,0);
		int label = data.second.argmax(1)(0, 0);
		for (int j = 0; j < 28; j++)
		{
			for (int k = 0; k < 28; k++)
			{
				int pixel = (int)(data.first(0, j * 28 + k) * 255) / 30;
				if (pixel == 0) std::cout << ". ";
				else std::cout << pixel << " ";
			}
			std::cout << std::endl;
		}
		std::cout << "Pred: " << pred << std::endl;
		std::cout << "Label: " << label << std::endl;
	}
}

void initTest(void)
{
	Sequential He_model = build_model("He");
	Sequential Xavier_model = build_model("Xavier");
	Sequential Normal_model = build_model("Normal");
	Sequential Uniform_model = build_model("Uniform");

	CrossEntropy cross_entropy = CrossEntropy();

	int num_samples = 2000;
	const int num_epochs = 30;
	const int batch_size = 32;
	float losses[4][num_epochs] = { 0 };

	tensor input = load_mnist_data("C:\\Users\\user\\Desktop\\train-images-idx3-ubyte\\train-images-idx3-ubyte", num_samples);
	tensor label = load_mnist_labels("C:\\Users\\user\\Desktop\\train-labels-idx1-ubyte\\train-labels-idx1-ubyte", num_samples);
	DataLoader data_loader = DataLoader(input, label, batch_size);

	float sum_loss = 0;
	int num_correct = 0;
	time_t start_time = 0;
	for (int j = 0; j < 4; j++)
	{
		Sequential model;
		if (j == 0) model = He_model;
		else if (j == 1) model = Xavier_model;
		else if (j == 2) model = Normal_model;
		else if (j == 3) model = Uniform_model;
		SGD sgd(model.layers, 0.01);
		std::cout << "SGD optimizer created" << std::endl;
		for (int i = 0; i < num_epochs; i++)
		{
			start_time = clock();
			while (data_loader.has_next())
			{
				std::pair<tensor, tensor> data = data_loader.get();
				tensor output = model(data.first);
				float loss = cross_entropy(output, data.second);
				sum_loss += loss;
				tensor output_grad = cross_entropy.backward(output, data.second);
				model.backward(output_grad);
				sgd.step();
			}
			losses[j][i] = sum_loss / data_loader.get_num_batchs();
			data_loader.reset();
			std::cout << "Epoch: " << i << " - " << (clock() - start_time) / 1000 << "s " << "- average loss: " << losses[j][i] << std::endl;
			num_correct = 0;
			sum_loss = 0;
		}
	}

	std::ofstream file("losses.csv");
	if (file.is_open()) {
		file << "Epoch,He,Xavier,Normal,Uniform\n";
		for (int i = 0; i < num_epochs; i++) {
			file << i << "," << losses[0][i] << "," << losses[1][i] << "," << losses[2][i] << "," << losses[3][i] << "\n";
		}
		file.close();
		std::cout << "Losses saved to losses.csv" << std::endl;
	}
	else {
		std::cerr << "Error opening file for writing!" << std::endl;
	}
}

int main() {
	mnistTest();
	return 0;
}