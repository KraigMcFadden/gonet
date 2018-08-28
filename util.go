package gonet

import (
	"math"
	"math/rand"
)

type ActivationFunction func(Vector)(Vector)

func random(a, b float64) float64 {
	return (b-a)*rand.Float64() + a
}

func sigmoid(vec Vector) Vector {
	outVec := new(Vector).Init(len(vec), 0.0)
	for i := 0; i < len(vec); i++ {
		outVec[i] = 1.0 / (1.0 + math.Exp(-vec[i]))
	}
	return outVec
}

func dsigmoid(vec Vector) Vector {
	outVec := new(Vector).Init(len(vec), 0.0)
	for i := 0; i < len(vec); i++ {
		outVec[i] = vec[i] * (1.0 - vec[i])
	}
	return outVec
}
