package gonet

import (
	"math/rand"
)

func random(a, b float64) float64 {
	return (b-a)*rand.Float64() + a
}

func ternary(condition bool, a, b float64) float64 {
	if condition {
		return a
	} else {
		return b
	}
}