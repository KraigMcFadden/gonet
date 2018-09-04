package gonet

import (
    "math/rand"
)

func testSetup() (a, b []Vector) {

    // set the random seed to 0
    rand.Seed(0)

    // create the XOR representation pattern to train the network
    inputs := []Vector{
      Vector{0, 0},
      Vector{0, 1},
      Vector{1, 0},
      Vector{1, 1},
    }

    labels := []Vector{
        Vector{0},
        Vector{1},
        Vector{1},
        Vector{0},
    }

    return inputs, labels
}

func ExampleSimpleNeuralNet1() {

    inputs, labels := testSetup()

    // initialize the Neural Network;
    // the networks structure will contain:
    // 2 inputs, 2 hidden nodes and 1 output.
    ff := new(NeuralNet).Init([]int{2, 2, 1}, []*ActivationFunction{sigmoid, sigmoid})

    // train the network using the XOR patterns
    // the training will run for 100000 epochs
    // the learning rate is set to 0.1 and the momentum factor to 0.05
    // use true in the last parameter to receive reports about the learning error
    ff.Train(inputs, labels, 100000, 0.1, 0.05, false)

    // testing the network
    ff.Test(inputs, labels)
    
    // Output: 
    // [0 0] -> [0.009734444150547994]  :  [0]
    // [0 1] -> [0.9882879617006709]  :  [1]
    // [1 0] -> [0.9910735517645498]  :  [1]
    // [1 1] -> [0.008242910418753075]  :  [0]
}

func ExampleDeepNeuralNet1() {
    
    inputs, labels := testSetup()

    // initialize the Neural Network;
    // the networks structure will contain:
    // 2 inputs, 3 layers of 10 hidden nodes each, and 1 output.
    ff := new(NeuralNet).Init([]int{2, 10, 10, 10, 1}, []*ActivationFunction{sigmoid, sigmoid, sigmoid, sigmoid})

    // train the network using the XOR patterns
    // the training will run for 10000 epochs
    // the learning rate is set to 0.1 and the momentum factor to 0.05
    // use true in the last parameter to receive reports about the learning error
    ff.Train(inputs, labels, 10000, 0.1, 0.05, false)

    // testing the network
    ff.Test(inputs, labels)
    
    // Output: 
    // [0 0] -> [0.022419631397543112]  :  [0]
    // [0 1] -> [0.9783862862996013]  :  [1]
    // [1 0] -> [0.9807939551838102]  :  [1]
    // [1 1] -> [0.019339945283183776]  :  [0]
}

func ExampleSimpleNeuralNet2() {

    inputs, labels := testSetup()

    // initialize the Neural Network;
    // the networks structure will contain:
    // 2 inputs, 2 hidden nodes and 1 output.
    ff := new(NeuralNet).Init([]int{2, 2, 2, 1}, []*ActivationFunction{relu, relu, sigmoid})

    // train the network using the XOR patterns
    // the training will run for 100000 epochs
    // the learning rate is set to 0.1 and the momentum factor to 0.05
    // use true in the last parameter to receive reports about the learning error
    ff.Train(inputs, labels, 10000, 0.2, 0.1, false)

    // testing the network
    ff.Test(inputs, labels)
    
    // Output: 
    // [0 0] -> [0.00913066911363663]  :  [0]
    // [0 1] -> [0.9903112574310573]  :  [1]
    // [1 0] -> [0.9918604935035364]  :  [1]
    // [1 1] -> [0.009070492866325849]  :  [0]
}

func ExampleDeepNeuralNet2() {
    
    inputs, labels := testSetup()

    // initialize the Neural Network;
    // the networks structure will contain:
    // 2 inputs, 3 layers of 10 hidden nodes each, and 1 output.
    ff := new(NeuralNet).Init([]int{2, 10, 10, 10, 1}, []*ActivationFunction{relu, relu, relu, sigmoid})

    // train the network using the XOR patterns
    // the training will run for 10000 epochs
    // the learning rate is set to 0.1 and the momentum factor to 0.05
    // use true in the last parameter to receive reports about the learning error
    ff.Train(inputs, labels, 10000, 0.1, 0.05, false)

    // testing the network
    ff.Test(inputs, labels)
    
    // Output: 
    // [0 0] -> [0.005805367318120523]  :  [0]
    // [0 1] -> [0.9955214776599985]  :  [1]
    // [1 0] -> [0.9958918597655823]  :  [1]
    // [1 1] -> [0.004083067111701076]  :  [0]
}

func ExampleDeepNeuralNet3() {
    
    inputs, labels := testSetup()

    // initialize the Neural Network;
    // the networks structure will contain:
    // 2 inputs, 3 layers of 10 hidden nodes each, and 1 output.
    ff := new(NeuralNet).Init([]int{2, 10, 10, 10, 1}, []*ActivationFunction{tanh, tanh, tanh, sigmoid})

    // train the network using the XOR patterns
    // the training will run for 10000 epochs
    // the learning rate is set to 0.1 and the momentum factor to 0.05
    // use true in the last parameter to receive reports about the learning error
    ff.Train(inputs, labels, 10000, 0.1, 0.05, false)

    // testing the network
    ff.Test(inputs, labels)
    
    // Output: 
    // [0 0] -> [0.0060849555933774205]  :  [0]
    // [0 1] -> [0.9956038524591562]  :  [1]
    // [1 0] -> [0.9929953794347924]  :  [1]
    // [1 1] -> [0.005853368972294614]  :  [0]
}