#include "network.h"
#include <fstream>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>

Network::Network(int inputSize, int hiddenSize, int outputSize, int hiddenLayers)
{
	//populate the input layers neurons
	inputLayer.size = inputSize;
	for (int i = 0; i < inputSize; i++)
	{
		//each neuron of the input layers takes one input value
		inputLayer.layer.push_back(Perceptron(1));
		inputLayer.results.push_back(0.0f);
	}

	//The first layer of the hidden layers should have inputs equal to the inputLayers
	hiddenLayer.push_back(Layer());
	for (int i = 0; i < hiddenSize; i++)
	{
		//The hidden layers after the first should all have inputs equal to the size of the hidden layers 
		hiddenLayer[0].layer.push_back(Perceptron(inputSize));
		hiddenLayer[0].results.push_back(0.0f);
	}
	hiddenLayer[0].size = hiddenSize;


	//loop through and create the hidden layers
	hiddenLayerSize = hiddenLayers;
	for (int j = 1; j < hiddenLayers; j++)
	{
		hiddenLayer.push_back(Layer());
		hiddenLayer[j].size = hiddenSize;

		//populate the hidden layers with neurons
		for (int i = 0; i < hiddenSize; i++)
		{
			//The hidden layers after the first should all have inputs equal to the size of the hidden layers 
			hiddenLayer[j].layer.push_back(Perceptron(hiddenSize));
			hiddenLayer[j].results.push_back(0.0f);

		}
	}

	//populate the output layers neurons
	for (int i = 0; i < outputSize; i++)
	{
		//each neuron of the input layers takes one input value
		outputLayer.layer.push_back(Perceptron(hiddenSize));
		outputLayer.results.push_back(0.0f);
	}
	outputLayer.size = outputSize;

	esum = INFINITY;
	lastEsum = 0.0f;

	//default target esum;
	target = 1.0f;

	//default loop counter 
	loopCounter = 50000;
}

//loading an existing neural network
Network::Network(std::string name)
{
	std::ifstream file(name);
	std::string line;

	int settings[7] = { 0 };

	std::getline(file, line);
	int counter = 0;
	while (line.size() > 1)
	{
		//string manipulation
		int pos = line.find(",");
		std::string sub = line.substr(0, pos);
		line = line.substr(pos + 1);

		//convert the string to a number
		settings[counter] = std::stoi(sub);
		counter++;
	}

	std::getline(file, line);
	inputLayer.size = settings[0];
	for (int i = 0; i < inputLayer.size; i++)
	{

		int pos = line.find(";");
		std::string sub = line.substr(0, pos);
		line = line.substr(pos + 1);

		inputLayer.layer.push_back(Perceptron(settings[1]));
		//set the feature weights
		double* data = new double[settings[1]];
		for (int j = 0; j < settings[1]; j++)
		{
			int pos2 = sub.find(",");
			std::string value = sub.substr(0, pos2);
			sub = sub.substr(pos2);
			data[j] = std::stod(value);
		}
		inputLayer.layer[i].SetWeights(data);
		inputLayer.results.push_back(0.0);

		delete[] data;
	}


	hiddenLayerSize = settings[4];
	for (int k = 0; k < hiddenLayerSize; k++)
	{
		hiddenLayer.push_back(Layer());
	}
	Layer* layers = hiddenLayer.data();

	int size = settings[3];
	for (int k = 0; k < hiddenLayerSize; k++)
	{
		std::getline(file, line);
		layers->size = settings[2];
		

		for (int i = 0; i < layers->size; i++)
		{

			int pos = line.find(";");
			std::string sub = line.substr(0, pos);
			line = line.substr(pos + 1);

			layers->layer.push_back(Perceptron(size));
			//set the feature weights
			double* data = new double[size];
			for (int j = 0; j < size; j++)
			{
				int pos2 = sub.find(",");
				std::string value = sub.substr(0, pos2);
				sub = sub.substr(pos2 + 1);
				data[j] = std::stod(value);
			}
			size = settings[2];
			layers->layer[i].SetWeights(data);
			layers->results.push_back(0.0);
			delete[] data;
		}
		layers++;
	}


	std::getline(file, line);
	outputLayer.size = settings[5];
	for (int i = 0; i < outputLayer.size; i++)
	{

		int pos = line.find(";");
		std::string sub = line.substr(0, pos);
		line = line.substr(pos + 1);

		outputLayer.layer.push_back(Perceptron(settings[6]));
		//set the feature weights
		double* data = new double[settings[6]];
		for (int j = 0; j < settings[6]; j++)
		{
			int pos2 = sub.find(",");
			std::string value = sub.substr(0, pos2);
			sub = sub.substr(pos2 + 1);
			data[j] = std::stod(value);
		}
		outputLayer.layer[i].SetWeights(data);
		outputLayer.results.push_back(0.0);
		delete[] data;
	}



}

//TODO: save function for neural nets
void Network::saveToFile(std::string name, bool overwrite)
{
	std::string fileName = name + ".txt";

	if (overwrite == false)
	{
		int counter = 0;
		//will loop until a file is found that doesn't exist
		while (std::ifstream(fileName))
		{
			counter++;
			fileName = name + std::to_string(counter) + ".txt";
		}
	}

	std::ofstream file(fileName);
	//csv file
	//there be 3 pieces
	// input, hidden, output
	// Isize, hSize, hCount, Osize
	// one line of IsizeWeights + bias
	// several lines in accordance to hCount of the number of hidden layers
	// one line for each of HsizeWeights + bias
	// one line for OsizeWeights + bias


	file << std::setprecision(19);

	//file settings setup
	file << std::to_string(inputLayer.size) << ",";
	file << std::to_string(inputLayer.layer[0].getFeatureVectorSize()) << ",";

	file << std::to_string(hiddenLayer[0].size) << ",";
	file << std::to_string(hiddenLayer[0].layer[0].getFeatureVectorSize()) << ",";
	file << std::to_string(hiddenLayerSize) << ",";

	file << std::to_string(outputLayer.size) << ",";
	file << std::to_string(outputLayer.layer[0].getFeatureVectorSize()) << ",";
	file << std::endl;


	Perceptron* data = inputLayer.layer.data();
	for (int i = 0; i < inputLayer.size; i++, data++)
	{
		for (int j = 0; j < data->getFeatureVectorSize(); j++)
		{
			file << data->getWeight(j);
			file << ",";
		}
		file << ";";
	}
	file << std::endl;

	Layer* layers = hiddenLayer.data();
	for (int i = 0; i < hiddenLayerSize; i++)
	{
		data = layers->layer.data();
		for (int j = 0; j < layers->size; j++, data++)
		{
			for (int l = 0; l < data->getFeatureVectorSize(); l++)
			{
				file << data->getWeight(l);
				file << ",";
			}
			file << ";";
		}
		layers++;
		file << std::endl;
	}

	data = outputLayer.layer.data();
	for (int i = 0; i < outputLayer.size; i++, data++)
	{
		for (int j = 0; j < data->getFeatureVectorSize(); j++)
		{
			file << data->getWeight(j);
			file << ",";
		}
		file << ";";
	}

	file.close();
}


void Network::input(double* inputs)
{
	for (int i = 0; i < inputLayer.size; i++)
	{
		inputLayer.results[i] = inputLayer.layer[i].Evaluate(inputs);
		inputs++;
	}


	double * data = inputLayer.results.data();
	Layer* current = hiddenLayer.data();

	for (int i = 0; i < hiddenLayerSize; i++)
	{
		for (int j = 0; j < current->size; j++)
		{
			current->results[j] = current->layer[j].Evaluate(data);
		}
		data = current->results.data();
		current++;
	}

	data = hiddenLayer[hiddenLayerSize - 1].results.data();
	for (int i = 0; i < outputLayer.size; i++)
	{
		outputLayer.results[i] = outputLayer.layer[i].Evaluate(data);
	}

}

Network::~Network()
{
	inputLayer.layer.clear();
	inputLayer.results.clear();

	for (int i = 0; i < hiddenLayer.size(); i++)
	{
		hiddenLayer[i].layer.clear();
		hiddenLayer[i].results.clear();
	}


	outputLayer.layer.clear();
	outputLayer.results.clear();
}


double Network::getFinalResult()
{
	double result;
	for (int i = 0; i < outputLayer.size; i++)
	{
		result = outputLayer.results[i];
	}
	return result;
}

void Network::IncrementRandomizeInputWeights(double min, double max)
{
	for (int i = 0; i < inputLayer.size; i++)
	{
		inputLayer.layer[i].AddRandomToWeight(min, max);
	}
}

void Network::IncrementRandomizeHiddenWeights(double min, double max)
{
	Layer* current = hiddenLayer.data();
	for (int i = 0; i < hiddenLayerSize; i++)
	{
		for (int j = 0; j < current->size; j++)
		{
			current->layer[j].AddRandomToWeight(min, max);
		}
		current++;
	}
}

void Network::IncrementRandomizeOutputWeights(double min, double max)
{
	for (int i = 0; i < outputLayer.size; i++)
	{
		outputLayer.layer[i].AddRandomToWeight(min, max);
	}
}

void Network::randomizeInputWeights(double min, double max)
{
	for (int i = 0; i < inputLayer.size; i++)
	{
		inputLayer.layer[i].AddRandomToWeight(min, max);
	}
}

void Network::randomizeOutputWeights(double min, double max)
{
	for (int i = 0; i < outputLayer.size; i++)
	{
		outputLayer.layer[i].RandomizeWeights(min, max);
	}
}

void Network::randomizeHiddenWeights(double min, double max)
{
	Layer* current = hiddenLayer.data();
	for (int i = 0; i < hiddenLayerSize; i++)
	{
		for (int j = 0; j < current->size; j++)
		{
			current->layer[j].RandomizeWeights(min, max);
		}
		current++;
	}
}

void Network::revertWeights()
{
	for (int i = 0; i < outputLayer.size; i++)
	{
		outputLayer.layer[i].revertRandomWeight();
	}

	Layer* current = hiddenLayer.data();
	for (int i = 0; i < hiddenLayerSize; i++)
	{
		for (int j = 0; j < current->size; j++)
		{
			current->layer[j].revertRandomWeight();
		}
		current++;
	}

	for (int i = 0; i < inputLayer.size; i++)
	{
		inputLayer.layer[i].revertRandomWeight();
	}

}