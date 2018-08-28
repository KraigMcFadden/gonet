package gonet

import (
	"math"
	"math/rand"
)

type ActivationFunction func(Vector)

func random(a, b float64) float64 {
	return (b-a)*rand.Float64() + a
}

func sigmoid(vec Vector) float64 {
	outVec := new(Vector).Init(vec.length, 0.0)
	for i := 0; i < vec.length; i++ {
		outVec[i] = 1.0 / (1.0 + math.Exp(-vec[i]))
	}
	return outVec
}

func dsigmoid(vec Vector) float64 {
	outVec := new(Vector).Init(vec.length, 0.0)
	for i := 0; i < vec.length; i++ {
		outVec[i] = vec[i] * (1.0 - vec[i])
	}
	return outVec
}
