// Package gonet provides a basic neural network

package gonet

import (
	"fmt"
	"log"
	"math"
	//"strconv"
)

// NeuralNet struct is used to represent a simple neural network
type NeuralNet struct {
    numNodes    []int          // Number of input, hidden and output nodes
	zs          []Vector       // weight times input
	alphas      []Vector       // activation(z)
	activations []ActivationFunction         // the activation function at each layer
	weights     []Matrix       // Weights for each layer
	changes     []Matrix       // Last change in weights for momentum

	numLayers   int            // helpful value to replace numNodes.length
}

/*
Initialize the neural network;

nodesPerLayer is an array of how many nodes should go into each layer
*/
func (nn *NeuralNet) Init(nodesPerLayer []int) *NeuralNet {
	
	layers := len(nodesPerLayer)
	nn.numLayers = layers

	nn.numNodes    = make([]int, layers)
	nn.zs          = make([]Vector, layers)
	nn.alphas      = make([]Vector, layers)
	nn.activations = make([]ActivationFunction, layers)
	nn.weights     = make([]Matrix, layers)
	nn.changes     = make([]Matrix, layers)

	for i := 0; i < layers; i++ {
		nn.numNodes[i] = nodesPerLayer[i]

		if i < layers - 1 {
			nn.zs[i] = new(Vector).Init(nodesPerLayer[i] + 1, 1.0)  // add one for bias term
			nn.alphas[i] = new(Vector).Init(nodesPerLayer[i] + 1, 1.0)
		} else {
			nn.zs[i] = new(Vector).Init(nodesPerLayer[i], 0.0)  // no bias term on output layer
			nn.alphas[i] = new(Vector).Init(nodesPerLayer[i], 0.0)
		}

		if i > 0 {
			nn.activations[i] = sigmoid
			nn.weights[i] = new(Matrix).Init(len(nn.alphas[i]), len(nn.alphas[i - 1])).RandomFill()
			nn.changes[i] = new(Matrix).Init(len(nn.alphas[i]), len(nn.alphas[i - 1]))
		}
	}

	return nn;
}

/*
The Update method is used to activate the Neural Network.

Given an array of inputs, it returns an array, of length equivalent of number of outputs, with values ranging from 0 to 1.
*/
func (nn *NeuralNet) Update(inputs Vector) Vector {
	if len(inputs) != nn.numNodes[0] {
		log.Fatal("Error: wrong number of inputs")
	}

	for n := 0; n < nn.numLayers; n++ {

		// this is the input layer
		if n == 0 {
			for i := 0; i < nn.numNodes[n]; i++ {
				nn.zs[0][i] = inputs[i]
				nn.alphas[0][i] = inputs[i]
			}
			continue
		}

		nn.zs[n] = nn.weights[n].Apply(nn.zs[n - 1])
		nn.alphas[n] = nn.activations[n](nn.zs[n])

		// // set biases back to 1 if they were messed up
		// if n < nn.numLayers - 1 {
		// 	nn.zs[n][nn.numNodes[n]] = 1.0
		// 	nn.alphas[n][nn.numNodes[n]] = 1.0
		// }
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

	outputDeltas := dsigmoid(nn.alphas[outLayer]).Mult(labels.Sub(nn.alphas[outLayer]))

	// for i := 0; i < numOutputs; i++ {
	// 	outputDeltas[i] = dsigmoid(nn.OutputActivations[i]) * (labels[i] - nn.OutputActivations[i])
	// }

	epsilons := nn.weights[outLayer].ReverseApply(outputDeltas)
	hiddenDeltas := dsigmoid(nn.alphas[outLayer - 1]).Mult(epsilons)

	// for i := 0; i < nn.NHiddens; i++ {
	// 	for j := 0; j < nn.NOutputs; j++ {
	// 		change := outputDeltas[j] * nn.HiddenActivations[i]
	// 		nn.OutputWeights[i][j] = nn.OutputWeights[i][j] + eta*change + mFactor*nn.OutputChanges[i][j]
	// 		nn.OutputChanges[i][j] = change
	// 	}
	// }

	momentum := nn.changes[outLayer].Scale(mFactor)
	nn.changes[outLayer] = nn.alphas[outLayer - 1].Cross(outputDeltas)
	nn.weights[outLayer] = nn.weights[outLayer].Add(nn.changes[outLayer].Scale(eta)).Add(momentum)

	// for i := 0; i < nn.NInputs; i++ {
	// 	for j := 0; j < nn.NHiddens; j++ {
	// 		change := hiddenDeltas[j] * nn.InputActivations[i]
	// 		nn.InputWeights[i][j] = nn.InputWeights[i][j] + eta*change + mFactor*nn.InputChanges[i][j]
	// 		nn.InputChanges[i][j] = change
	// 	}
	// }

	momentum = nn.changes[outLayer - 1].Scale(mFactor)
	nn.changes[outLayer - 1] = nn.alphas[outLayer - 2].Cross(hiddenDeltas)
	nn.weights[outLayer - 1] = nn.weights[outLayer - 1].Add(nn.changes[outLayer - 1].Scale(eta)).Add(momentum)

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
func (nn *NeuralNet) Train(patterns [][][]float64, iterations int, eta, mFactor float64, debug bool) []float64 {
	errors := make([]float64, iterations)

	for i := 0; i < iterations; i++ {
		var e float64
		for _, p := range patterns {
			nn.Update(p[0])

			tmp := nn.BackPropagate(p[1], eta, mFactor)
			e += tmp
		}

		errors[i] = e

		if debug && i%1000 == 0 {
			fmt.Println(i, e)
		}
	}

	return errors
}

func (nn *NeuralNet) Test(patterns [][][]float64) {
	for _, p := range patterns {
		fmt.Println(p[0], "->", nn.Update(p[0]), " : ", p[1])
	}
}
