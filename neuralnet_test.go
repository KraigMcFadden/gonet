package gonet

import (
    "math/rand"
)

func ExampleSimpleNeuralNet() {
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

    // initialize the Neural Network;
    // the networks structure will contain:
    // 2 inputs, 2 hidden nodes and 1 output.
    ff := new(NeuralNet).Init([]int{2, 2, 1})

    // train the network using the XOR patterns
    // the training will run for 100000 epochs
    // the learning rate is set to 0.1 and the momentum factor to 0.05
    // use true in the last parameter to receive reports about the learning error
    ff.Train(inputs, labels, 100000, 0.1, 0.05, false)

    // testing the network
    ff.Test(inputs, labels)

    // predicting a value
    updateInputs := Vector{1, 1}
    ff.Update(updateInputs)
    
    // Output: 
    // [0 0] -> [0.009734444150547994]  :  [0]
    // [0 1] -> [0.9882879617006709]  :  [1]
    // [1 0] -> [0.9910735517645498]  :  [1]
    // [1 1] -> [0.008242910418753075]  :  [0]
}

func ExampleDeepNeuralNet() {
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

    // initialize the Neural Network;
    // the networks structure will contain:
    // 2 inputs, 3 layers of 10 hidden nodes each, and 1 output.
    ff := new(NeuralNet).Init([]int{2, 10, 10, 10, 1})

    // train the network using the XOR patterns
    // the training will run for 10000 epochs
    // the learning rate is set to 0.1 and the momentum factor to 0.05
    // use true in the last parameter to receive reports about the learning error
    ff.Train(inputs, labels, 10000, 0.1, 0.05, false)

    // testing the network
    ff.Test(inputs, labels)

    // predicting a value
    updateInputs := Vector{1, 1}
    ff.Update(updateInputs)
    
    // Output: 
    // [0 0] -> [0.022419631397543112]  :  [0]
    // [0 1] -> [0.9783862862996013]  :  [1]
    // [1 0] -> [0.9807939551838102]  :  [1]
    // [1 1] -> [0.019339945283183776]  :  [0]
}