#include "Perceptron.h"

#include <cmath>
#include <stdlib.h>

double SigmoidFunction(double val)
{
	const double e = 2.71828182845904523536f;

	return 1.0f / (1.0f + std::powf(e, -val));
}

double RandomRange(double min, double max)
{
	return min + ((max - min) * rand()) / (RAND_MAX + 1.0f);
}

void Perceptron::setFeatureVectorSize(unsigned int _featureVectorSize)
{
	//cleanup weights
	delete[] weights;
	delete[] oldWeights;

	featureVectorSize = _featureVectorSize;
	weights = new double[featureVectorSize];
	oldWeights = new double[featureVectorSize];

	for (unsigned int i = 0; i < featureVectorSize; ++i)
	{
		weights[i] = 0.0f;
		oldWeights[i] = 0.0f;
	}
}

Perceptron::Perceptron(unsigned int _featureVectorSize)
	: featureVectorSize(_featureVectorSize)
{
	weights = new double[featureVectorSize];
	oldWeights = new double[featureVectorSize];

	for (unsigned int i = 0; i < featureVectorSize; ++i)
	{
		weights[i] = 0.0f;
		oldWeights[i] = 0.0f;
	}
}

Perceptron::Perceptron(const Perceptron& other)
	: featureVectorSize(other.featureVectorSize)
	, bias(other.bias)
{
	weights = new double[featureVectorSize];
	oldWeights = new double[featureVectorSize];

	// Copy the values from the other Perceptron.
	for (unsigned int i = 0; i < featureVectorSize; ++i)
	{
		weights[i] = other.weights[i];
		oldWeights[i] = other.weights[i];
	}
}

Perceptron::~Perceptron()
{
	delete[] weights;
	delete[] oldWeights;
}

Perceptron Perceptron::Crossover(const Perceptron& parent1, const Perceptron& parent2)
{
	// Naturally, we are assuming parent1.featureVectorSize == parent2.featureVectorSize.
	Perceptron result(parent1.featureVectorSize);

	// Average all the values between the parents.
	for (unsigned int i = 0; i < result.featureVectorSize; ++i)
	{
		result.weights[i] = (parent1.weights[i] + parent2.weights[i]) / 2.0f;
	}

	result.bias = (parent1.bias + parent2.bias) / 2.0f;

	return result;
}

double Perceptron::Evaluate(const double* featureVector) const
{
	double result = 0.0f;
	for (unsigned int i = 0; i < featureVectorSize; ++i)
	{
		result += featureVector[i] * weights[i];
	}

	result += bias;

	return SigmoidFunction(result);
}

void Perceptron::SetWeights(const double* _weights)
{
	for (unsigned int i = 0; i < featureVectorSize; ++i)
	{
		weights[i] = _weights[i];
		oldWeights[i] = _weights[i];
	}
}

void Perceptron::RandomizeWeights(double min, double max)
{
	for (unsigned int i = 0; i < featureVectorSize; ++i)
	{
		weights[i] = RandomRange(min, max); 
		oldWeights[i] = weights[i];
	}
}

void Perceptron::AddRandomToWeight(double min, double max)
{
	for (unsigned int i = 0; i < featureVectorSize; ++i)
	{
		oldWeights[i] = weights[i];
		weights[i] += RandomRange(min, max); 
	}
}

void Perceptron::revertRandomWeight()
{
	for (unsigned int i = 0; i < featureVectorSize; ++i)
	{
		weights[i] = oldWeights[i];
	}
}


void Perceptron::RandomizeBias(double min, double max)
{
	bias = RandomRange(min, max);
}

double Perceptron::getWeight(int index)
{
	return weights[index];
}

unsigned int Perceptron::getFeatureVectorSize()
{
	return featureVectorSize;
}