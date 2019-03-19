#pragma once

#include "Perceptron.h"
#include <vector>


struct Layer
{
	//the layer
	std::vector<Perceptron> layer;
	//the results of the layer
	std::vector<double> results;
	//size of the vectors 
	int size;
};

class Network
{
public:
	//creating a new neural network
	Network(int inputSize, int hiddenSize, int outputSize,  int hiddenLayers = 1);

	//loading an existing neural network
	Network(std::string name);

	~Network();

	void input(double* inputs);
	double getFinalResult();


	void revertWeights();

	void randomizeInputWeights(double min, double max);
	void randomizeOutputWeights(double min, double max);
	void randomizeHiddenWeights(double min, double max);

	void IncrementRandomizeInputWeights(double min, double max);
	void IncrementRandomizeOutputWeights(double min, double max);
	void IncrementRandomizeHiddenWeights(double min, double max);



	//TODO: save function for neural nets
	void saveToFile(std::string name = "neuralnet", bool overwrite = false);

	//TODO: function to edit layers
	void setInputLayer();
	void editHiddenLayer();
	void addHiddenLayer();
	void setOutputLayer();

private:

	double esum;
	double lastEsum;
	double target;
	int loopCounter;

	//The assumption is that the input layer feeds into the first layer of the hidden layer
	//then each hidden layer feeds into the next hidden layer
	//the final hiddenlayer pipes to outputLayer
	Layer inputLayer;

	int hiddenLayerSize;
	std::vector<Layer> hiddenLayer;
	Layer outputLayer;
};

