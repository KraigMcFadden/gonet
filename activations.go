package gonet

import (
	"math"
)

type ActivationFunction struct {
	Id string
	f  func(float64)(float64)
	df func(float64)(float64)
}

func (f *ActivationFunction) F(v Vector) Vector {
	return v.Map(f.f)
}

func (f *ActivationFunction) Df(v Vector) Vector {
	return v.Map(f.df)
}

var sigmoid *ActivationFunction = getActivation("sigmoid")
var tanh *ActivationFunction = getActivation("tanh")
var relu *ActivationFunction = getActivation("relu")

func getActivation(id string) *ActivationFunction {
	activation := new(ActivationFunction)
	activation.Id = id
	switch id {
		case "sigmoid": 
			activation.f = func(x float64) float64 { return 1.0 / (1.0 + math.Exp(-x)) }
			activation.df = func(x float64) float64 {
					ex := math.Exp(x)
					return ex / ((1.0 + ex) * (1.0 + ex))
				}
			break
		case "tanh":
			activation.f = func(x float64) float64 { return math.Tanh(x) }
			activation.df = func(x float64) float64 {
					tanhx := math.Tanh(x)
					return 1.0 - (tanhx * tanhx)
				}
			break
		case "relu":
			activation.f = func(x float64) float64 { return ternary(x > 0.0, x, 0.2*x) }
			activation.df = func(x float64) float64 { return ternary(x > 0.0, 1.0, 0.2) }
			break
		default:
			activation = getActivation("sigmoid")
	}
	return activation
}