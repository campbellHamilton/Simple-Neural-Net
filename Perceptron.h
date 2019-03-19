#ifndef _PERCEPTRON_H_
#define _PERCEPTRON_H_

class Perceptron
{
public:
	Perceptron(unsigned int featureVectorSize = 1);
	Perceptron(const Perceptron& other);
	~Perceptron();

	void setFeatureVectorSize(unsigned int featureVectorSize);
	unsigned int getFeatureVectorSize();

	// Naively mixes two Perceptrons together to generate a new Perceptron which is a mix of the two parents.
	// This can be used for Genetic Algorithms.
	static Perceptron Crossover(const Perceptron& parent1, const Perceptron& parent2);

	// Evaluates the full mathematical expression of the Perceptron:
	// val = activationFunc(Sum(feature * weight) +- bias)
	// FeatureVector must be a pointer to an array of doubles with length 'featureVectorSize'.
	double Evaluate(const double* featureVector) const;

	// Generates random values for each weight and the bias.
	void RandomizeWeights(double min = -2.0f, double max = 2.0f);
	void AddRandomToWeight(double min, double max);
	void revertRandomWeight();
	void RandomizeBias(double min, double max);
	double getWeight(int index);

	// 'weights' must be a pointer to an array of doubles with length 'featureVectorSize'.
	void SetWeights(const double* weights);

	double bias = 0.0f;

private:
	// Dynamically allocated array of weights.
	double* weights;

	double* oldWeights;

	// Remembers the size of the allocated arrays.
	unsigned int featureVectorSize;
};

#endif