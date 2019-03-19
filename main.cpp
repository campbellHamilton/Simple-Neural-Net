#include <GL/gl3w.h>
#include <GLFW/glfw3.h> // GLFW helper library
#include <GLM/glm.hpp>
#include <GLM/gtc/matrix_transform.hpp> // for glm::ortho
using namespace glm;

// IMGUI
#include <imgui.h>
#include <imgui_impl_glfw_gl3.h>

#include <iostream> // Used for 'cout'
#include <stdio.h>  // Used for 'printf'
#include <time.h>   // Used to seed the rand
#include <vector>   // Used for std::vector<Model>
#include <fstream>

#include <string>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include "network.h"
#include "Perceptron.h"

/*---------------------------- Variables ----------------------------*/
// GLFW window
GLFWwindow* window;
int width = 1280;
int height = 720;
bool mouseDownLeft = false;
bool mouseDownRight = false;

// AI Variables
const bool USE_CUSTOM_PERCEPTRON_VALUES = true; /**  EDIT ME!!  **/
enum Features
{
	PIXEL,
	COUNT
};

enum Outputs
{
	NONE,
	Circle,
	NotACircle
};

Outputs result = Outputs::NONE;
double perceptronOutput = 0.0f;

// Feature vector stuff
vec2 directionAccumulation = vec2(0.0f, 0.0f);
vec2 maxBounds = vec2(0.0f, 0.0f);
vec2 minBounds = vec2(0.0f, 0.0f);
vec2 startPoint = vec2(0.0f, 0.0f);
vec2 endPoint = vec2(0.0f, 0.0f);
vec2 mousePosition = vec2(0.0f, 0.0f);


const int size = 1024;


//old main

int oldMain()
{
	srand((unsigned int)time(NULL));

	Network network(size, 10, 1);

	Perceptron inputLayer[size]{};
	Perceptron hiddenLayer[10]{};
	Perceptron outputLayer(10);


	for (int i = size - 1; i >= 0; i--)
	{
		inputLayer[i].setFeatureVectorSize(Features::COUNT);
		inputLayer[i].RandomizeWeights(-1.0f, 1.0f);
	}

	for (int i = 9; i >= 0; i--)
	{
		hiddenLayer[i].setFeatureVectorSize(size);
		hiddenLayer[i].RandomizeWeights(-0.5f, 0.5f);
	}

	outputLayer.setFeatureVectorSize(10);
	outputLayer.RandomizeWeights(-1.0f, 1.0f);


	std::ofstream ofile;
	ofile.open("..\\src\\Assignments\\final project\\output.txt");

	std::ifstream file;
	file.open("..\\src\\Assignments\\final project\\testing.txt");

	double inputLayerResults[1024]{ 0.0f };
	double hiddenLayerResults[10]{ 0.0f };


	if (file.is_open())
	{
		double lastEsum = INFINITE;
		double esum = 2.0f;
		int loopCounter = 0;

		double target = 0.01f;

		while (esum > target)
		{
			loopCounter++;
			esum = 0;

			std::string line;

			std::ifstream file;
			file.open("..\\src\\Assignments\\final project\\testing.txt");
			int counter;
			for (int i = 20; i >= 0; i--)
			{
				counter = 0;

				for (int j = 31; j >= 0; j--)
				{
					std::getline(file, line);
					for (int l = line.size() - 1; l >= 0; l--)
					{
						double value = line.at(l) - 48; //48 is zero in chars
						inputLayerResults[counter] = inputLayer[counter].Evaluate(&value);
						counter++;
					}
				}


				for (int j = 9; j >= 0; j--)
				{
					hiddenLayerResults[j] = hiddenLayer[j].Evaluate(inputLayerResults);


					//std::ostringstream streamObj;
					//streamObj << std::setprecision(17);
					//streamObj << hiddenLayerResults[j];
					//std::string preciseString = streamObj.str();
					//std::cout << preciseString << ", ";
				}
				double result = outputLayer.Evaluate(hiddenLayerResults);
				//std::ostringstream streamObj;
				//streamObj << std::setprecision(17);
				//streamObj << result;
				//std::string preciseString = streamObj.str();
				//std::cout << preciseString;

				std::getline(file, line);
				int value = std::stoi(line);
				//std::cout << " and it was actually a " + line << std::endl;
				//if it was not the target value
				double elet;
				//not 0
				if (value >= 1)
				{
					//a high result means its a 0. therefore the smaller it is here the less error
					elet = result;
				}
				//is 0
				else
				{
					//a low result here means an error
					//the larger result is the smaller the value added to elet is
					//We want more emphasis on getting the zero right.
					elet = (result - 1) * 3;
				}
				esum += elet * elet;
			}


			printf("%F", esum);
			if (lastEsum < esum)
			{
				printf("it got worse uh oh");
				for (int i = size - 1; i >= 0; i--)
				{
					inputLayer[i].revertRandomWeight();
				}

				for (int i = 9; i >= 0; i--)
				{
					hiddenLayer[i].revertRandomWeight();
				}
			}
			printf("\n");

			for (int i = size - 1; i >= 0; i--)
			{
				inputLayer[i].AddRandomToWeight(-0.05, 0.05);
			}

			for (int i = 9; i >= 0; i--)
			{
				hiddenLayer[i].AddRandomToWeight(-0.001, 0.001);
			}
			lastEsum = esum;
			file.close();
		}
	}

	std::ostringstream streamObj;
	std::ostringstream output;
	output << std::setprecision(2);
	streamObj << std::setprecision(19);

	output << "input layer\n";

	for (int i = size - 1; i >= 0; i--)
	{
		double value = inputLayer[i].getWeight(0);
		streamObj << value;
		output << std::round(value * 10);
		streamObj << ",";
	}

	output << "\n\n";
	streamObj << std::endl;
	streamObj << "new \n";

	for (int i = 9; i >= 0; i--)
	{
		for (int j = size - 1; j >= 0; j--)
		{
			double value = hiddenLayer[i].getWeight(j);
			streamObj << value;
			streamObj << ",";
		}
		streamObj << std::endl;
	}

	streamObj << "new \n";

	for (int i = 9; i >= 0; i--)
	{
		double value = outputLayer.getWeight(i);
		streamObj << value;
		streamObj << ",";
	}

	ofile << streamObj.str();
	ofile.close();

	// setup the 
	file.close();
	file.open("..\\src\\Assignments\\final project\\testing.txt");

	for (int i = 945; i >= 0; i--)
	{
		std::string line;
		int counter = 0;

		streamObj.clear();

		for (int j = 31; j >= 0; j--)
		{
			std::getline(file, line);
			for (int l = line.size() - 1; l >= 0; l--)
			{
				double value = line.at(l) - 48; //48 is zero in chars
				inputLayerResults[counter] = inputLayer[counter].Evaluate(&value);
				counter++;
			}
		}

		for (int j = 9; j >= 0; j--)
		{
			hiddenLayerResults[j] = hiddenLayer[j].Evaluate(inputLayerResults);
		}

		double result = outputLayer.Evaluate(hiddenLayerResults);
		std::ostringstream streamObj;
		std::ostringstream output;
		streamObj << std::setprecision(17);
		std::string preciseString = streamObj.str();
		std::getline(file, line);
		int value = std::stoi(line);

		//its zero
		if (result > 0.5)
		{
			std::cout << "is zero \n";
			//is not actually a zero
			if (value >= 1)
			{
				std::cout << preciseString;
				std::cout << " failed and it was actually a " + line << std::endl;
			}
		}
		//is 1+
		else
		{
			//std::cout << "not zero ";
			//is actually a zero
			if (value == 0)
			{
				std::cout << preciseString;
				std::cout << " failed and it was actually a " + line << std::endl;
			}
		}
	}


	return 0;
}

//new class based main
int main()
{

	srand((unsigned int)time(NULL));

	Network network(size, 10, 1);


	network.randomizeHiddenWeights(-1.0f, 1.0f);
	network.randomizeInputWeights(-0.5f, 0.5f);
	network.randomizeOutputWeights(-1.0f, 1.0f);

	std::ifstream file;
	file.open("..\\src\\Assignments\\final project\\testing.txt");
	
	double data[1024];

	if (file.is_open())
	{
		double lastEsum = INFINITE;
		double esum = 2.0f;
		int loopCounter = 0;

		double target = 1.0f;

		while (esum > target)
		{
			loopCounter++;
			esum = 0;

			std::string line;

			std::ifstream file;
			file.open("..\\src\\Assignments\\final project\\testing.txt");
			int counter;
			for (int i = 20; i >= 0; i--)
			{
				counter = 0;
				for (int j = 31; j >= 0; j--)
				{
					std::getline(file, line);
					for (int l = line.size() - 1; l >= 0; l--)
					{
						data[counter] = line.at(l) - 48; //48 is zero in chars
						counter++;
					}
				}
				network.input(data);

				std::getline(file, line);
				int value = std::stoi(line);
				double result = network.getFinalResult();
				double elet;
				//not 0
				if (value >= 1)
				{
					//a high result means its a 0. therefore the smaller it is here the less error
					elet = result;
				}
				//is 0
				else
				{
					//a low result here means an error
					//the larger result is the smaller the value added to elet is
					//We want more emphasis on getting the zero right.
					elet = (result - 1) * 3;
				}
				esum += elet * elet;
			}


			printf("%F", esum);
			if (lastEsum < esum)
			{
				printf("it got worse uh oh");
				network.revertWeights();
			}
			printf("\n");


			network.IncrementRandomizeInputWeights(-0.05f, 0.05f);
			network.IncrementRandomizeHiddenWeights(-0.001f, 0.001f);
//			network.IncrementRandomizeOutputWeights(-0.05f, 0.05f);

			lastEsum = esum;
		}
	}

	file.close();
	file.open("..\\src\\Assignments\\final project\\testing.txt");

	std::ofstream ofile2("tester1.txt");
	std::ofstream ofile3("tester2.txt");
	network.saveToFile("testing", true);
	Network loaded("testing.txt");


	for (int i = 945; i >= 0; i--)
	{
		std::string line;
		int counter = 0;


		for (int j = 31; j >= 0; j--)
		{
			std::getline(file, line);
			for (int l = line.size() - 1; l >= 0; l--)
			{
				data[counter] = line.at(l) - 48; //48 is zero in chars
				counter++;
			}
		}
		network.input(data);
		loaded.input(data);


		double result = network.getFinalResult();
		double result2 = loaded.getFinalResult();
		std::ostringstream streamObj;
		streamObj << std::setprecision(17);
		streamObj << result;

		std::string preciseString = streamObj.str();
		std::getline(file, line);
		int value = std::stoi(line);

		//its zero
		if (result > 0.5)
		{
			ofile2 << "is zero \n";
			//is not actually a zero
			if (value >= 1)
			{
				ofile2 << preciseString;
				ofile2 << " failed and it was actually a " + line << std::endl;
			}
		}
		//is 1+
		else
		{
			ofile2 << "not zero ";
			//is actually a zero
			if (value == 0)
			{
				ofile2 << preciseString;
				ofile2 << " failed and it was actually a " + line << std::endl;
			}
		}
	}



	printf("\n loaded inputs \n");

	file.close();


	//
	//for (int i = 945; i >= 0; i--)
	//{
	//	std::string line;
	//	int counter = 0;


	//	for (int j = 31; j >= 0; j--)
	//	{
	//		std::getline(file, line);
	//		for (int l = line.size() - 1; l >= 0; l--)
	//		{
	//			data[counter] = line.at(l) - 48; //48 is zero in chars
	//			counter++;
	//		}
	//	}
	//	loaded.input(data);


	//	double result = loaded.getFinalResult();
	//	std::ostringstream streamObj;
	//	streamObj << std::setprecision(17);
	//	streamObj << result;

	//	std::string preciseString = streamObj.str();
	//	std::getline(file, line);
	//	int value = std::stoi(line);

	//	//its zero
	//	if (result > 0.5)
	//	{
	//		ofile3 << "is zero \n";
	//		//is not actually a zero
	//		if (value >= 1)
	//		{
	//			ofile3 << preciseString;
	//			ofile3 << " failed and it was actually a " + line << std::endl;
	//		}
	//	}
	//	//is 1+
	//	else
	//	{
	//		ofile3 << "not zero ";
	//		//is actually a zero
	//		if (value == 0)
	//		{
	//			ofile3 << preciseString;
	//			ofile3 << " failed and it was actually a " + line << std::endl;
	//		}
	//	}
	//}

	ofile2.close();
	ofile3.close();

	return 0;
}