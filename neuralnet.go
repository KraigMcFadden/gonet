// Package gonet provides a basic neural network

package gonet

import (
	"fmt"
	"log"
	"math"
)

// NeuralNet struct is used to represent a simple neural network
type NeuralNet struct {
    numNodes    []int          // Number of input, hidden and output nodes
	zs          []Vector       // weight times input
	alphas      []Vector       // activation(z)
	biases      []Vector       // bias values
	activations []*ActivationFunction         // the activation function at each layer
	weights     []Matrix       // Weights for each layer
	changes     []Matrix       // Last change in weights for momentum

	numLayers   int            // helpful value to replace len(nn.numNodes)
}

/*
Initialize the neural network;

nodesPerLayer is an array of how many nodes should go into each layer.
activations is the activation function for each layer - do not specify an activation for input layer.
*/
func (nn *NeuralNet) Init(nodesPerLayer []int, activations []*ActivationFunction) *NeuralNet {
	
	layers := len(nodesPerLayer)
	nn.numLayers = layers

	nn.numNodes    = make([]int, layers)
	nn.zs          = make([]Vector, layers)
	nn.alphas      = make([]Vector, layers)
	nn.biases      = make([]Vector, layers)
	nn.activations = make([]*ActivationFunction, layers)
	nn.weights     = make([]Matrix, layers)
	nn.changes     = make([]Matrix, layers)

	for i := 0; i < layers; i++ {

		nn.numNodes[i] = nodesPerLayer[i]
		nn.zs[i] = new(Vector).Init(nodesPerLayer[i], 0.0)
		nn.alphas[i] = new(Vector).Init(nodesPerLayer[i], 0.0)

		if i > 0 {
			nn.activations[i] = activations[i - 1]
			nn.biases[i] = new(Vector).Init(nodesPerLayer[i], 0.0).RandomFill()
			nn.weights[i] = new(Matrix).Init(nodesPerLayer[i], nodesPerLayer[i - 1]).RandomFill()
			nn.changes[i] = new(Matrix).Init(nodesPerLayer[i], nodesPerLayer[i - 1])
		}
	}

	return nn;
}

/*
The Update method is used to activate the Neural Network.

Given an array of inputs, it returns an array, of length equivalent of number of outputs, with values 
ranging from the min to the max of the activation function at the output layer.
*/
func (nn *NeuralNet) Update(inputs Vector) Vector {
	if len(inputs) != nn.numNodes[0] {
		log.Fatal("Error: wrong number of inputs")
	}

	// copy inputs
	for i := 0; i < len(inputs); i++ {
		nn.zs[0][i] = inputs[i]
		nn.alphas[0][i] = inputs[i]
	}

	// feedforward through layers
	for n := 1; n < nn.numLayers; n++ {
		nn.zs[n] = nn.weights[n].Apply(nn.alphas[n - 1]).Add(nn.biases[n])
		nn.alphas[n] = nn.activations[n].F(nn.zs[n])
	}

	return nn.alphas[nn.numLayers - 1]
}
/*
The BackPropagate method is used, when training the Neural Network,
to back propagate the errors from network activation.
*/
func (nn *NeuralNet) BackPropagate(labels Vector, eta, mFactor float64) float64 {
	outLayer := nn.numLayers - 1
	if len(labels) != nn.numNodes[outLayer] {
		log.Fatal("Error: wrong number of target values")
	}
	
	// compute deltas
	deltas := make([]Vector, nn.numLayers)
	deltas[outLayer] = nn.activations[outLayer].Df(nn.zs[outLayer]).Mult(nn.alphas[outLayer].Sub(labels))
	for n := outLayer; n - 1 > 0; n-- {
		epsilons := nn.weights[n].ReverseApply(deltas[n])
		deltas[n - 1] = nn.activations[n].Df(nn.zs[n - 1]).Mult(epsilons)
	}

	// adjust weights and biases
	for n := outLayer; n > 0; n-- {
		momentum := nn.changes[n].Scale(mFactor)
		nn.changes[n] = deltas[n].Cross(nn.alphas[n - 1])
		nn.weights[n] = nn.weights[n].Sub(nn.changes[n].Scale(eta)).Sub(momentum)
		nn.biases[n] = nn.biases[n].Sub(deltas[n].Scale(eta))
	}

	var e float64
	for i := 0; i < len(labels); i++ {
		e += 0.5 * math.Pow(labels[i] - nn.alphas[outLayer][i], 2)
	}
	return e
}
/*
This method is used to train the Network, it will run the training operation for 'iterations' times
and return the computed errors when training.
*/
func (nn *NeuralNet) Train(inputs, labels []Vector, iterations int, eta, mFactor float64, debug bool) []float64 {
	errors := make([]float64, iterations)

	for i := 0; i < iterations; i++ {
		var e float64
		for i := 0; i < len(inputs); i++ {
			nn.Update(inputs[i])

			tmp := nn.BackPropagate(labels[i], eta, mFactor)
			e += tmp
		}

		errors[i] = e

		if debug && i%1000 == 0 {
			fmt.Println(i, e)
		}
	}

	return errors
}

func (nn *NeuralNet) Test(inputs, labels []Vector) {
	for i := 0; i < len(inputs); i++ {
		fmt.Println(inputs[i], "->", nn.Update(inputs[i]), " : ", labels[i])
	}
}
