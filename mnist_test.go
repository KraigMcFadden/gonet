package gonet

import (
	"math/rand"
    "strconv"
    "fmt"
)
 
func ExampleSimpleMNISTNet() {

	// set the random seed to 0
    rand.Seed(0)

    trainInputs, trainLabels, testInputs, testLabels := mnistTestSetup()

    // initialize the Neural Network;
    // the networks structure will contain:
    // len(trainInputs) inputs, 10 hidden nodes and 10 outputs.
    ff := new(NeuralNet).Init([]int{len(trainInputs[0]), 10, len(trainLabels[0])}, []*ActivationFunction{sigmoid, sigmoid})

    // train the network using MNIST data
    // the training will run for 100 epochs
    // the learning rate is set to 0.1 and the momentum factor to 0.05
    // use true in the last parameter to receive reports about the learning error
    ff.Train(trainInputs, trainLabels, 1, 0.1, 0.05, false)

    // testing the network
    numCorrect := 0
    for i := 0; i < len(testInputs); i++ {
    	if pickOneHot(ff.Update(testInputs[i])) == pickOneHot(testLabels[i]) {
    		numCorrect++
    	}
    }
    fmt.Println(strconv.Itoa(numCorrect) + " correct out of " + strconv.Itoa(len(testInputs)))
    
    // Output: 
    // 8923 correct out of 10000
}