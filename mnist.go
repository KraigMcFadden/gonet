package gonet

import (
    "encoding/csv"
    "bufio"
    "io"
    "os"
    "log"
    "strconv"
    "fmt"
    "sync"
)

func parseCsv(filename string) (a, b []Vector) {
    datas := make([]Vector, 0)
    labels := make([]Vector, 0)

    f, err := os.Open(filename)
    if err != nil {
        log.Fatal(err)
    }

    r := csv.NewReader(bufio.NewReader(f))
    for {
        record, err := r.Read()
        if err == io.EOF {
            break
        }
        if err != nil {
            log.Fatal(err)
        }

        label := new(Vector).Init(10, 0)
        index, _ := strconv.Atoi(record[0])
        label[index] = 1
        labels = append(labels, label)


        data := new(Vector).Init(len(record) - 1, 0)
        for i := 0; i < len(data); i++ {
            val, _ := strconv.ParseFloat(record[i + 1], 64)
            data[i] = val / 255.0  // normalize
        }
        datas = append(datas, data)
    }
    return datas, labels
}

func mnistTestSetup() (a, b, c, d []Vector) {

    // set up inputs and labels
    dir, err := os.Getwd()
    if err != nil {
        log.Fatal(err)
    }
    var trainInputs, trainLabels, testInputs, testLabels []Vector

    var wg sync.WaitGroup
    wg.Add(2)
    go func() {
        defer wg.Done()
        trainInputs, trainLabels = parseCsv(dir + "/src/gonet/mnist_train.csv")
    }()
    go func() {
        defer wg.Done()
        testInputs, testLabels = parseCsv(dir + "/src/gonet/mnist_test.csv")
    }()

    wg.Wait()
    return trainInputs, trainLabels, testInputs, testLabels
}

func pickOneHot(vec Vector) int {
    maxVal := 0.0
    index := 0
    for i := 0; i < len(vec); i++ {
        if vec[i] > maxVal {
            maxVal = vec[i]
            index = i
        }
    }
    return index
}

func DeepMNISTNet() {

    trainInputs, trainLabels, testInputs, testLabels := mnistTestSetup()


    // initialize the Neural Network;
    // the networks structure will contain:
    // len(trainInputs) inputs, 10 hidden nodes and 10 outputs.
    ff := new(NeuralNet).Init([]int{len(trainInputs[0]), 10, 10, len(trainLabels[0])}, []*ActivationFunction{tanh, tanh, sigmoid})

    // train the network using MNIST data
    // the training will run for 100 epochs
    // the learning rate is set to 0.1 and the momentum factor to 0.05
    // use true in the last parameter to receive reports about the learning error
    ff.Train(trainInputs, trainLabels, 100, 0.01, 0.005, true)

    // testing the network
    numCorrect := 0
    for i := 0; i < len(testInputs); i++ {
    	if pickOneHot(ff.Update(testInputs[i])) == pickOneHot(testLabels[i]) {
    		numCorrect++
    	}
    }
    fmt.Println(strconv.Itoa(numCorrect) + " correct out of " + strconv.Itoa(len(testInputs)))
}