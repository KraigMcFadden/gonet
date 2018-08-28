package gonet

type Vector []float64

func (v *Vector) Init(length int, initialFill float64) *Vector {
	v = make([]float64, length)
	for i := 0; i < length; i++ {
		v[i] = fill
	}
	return v
}

func (v *Vector) Cross(vec Vector) *Matrix {
	outMatrix := new(Matrix).Init(len(v), len(vec))
	for i := 0; i < len(v); i++ {
	 	for j := 0; j < len(vec); j++ {
	 		outMatrix[i][j] = v[i] * vec[j]
	 	}
	}
	return outMatrix
}

func (v *Vector) Mult(vec Vector) *Vector {
	outVec := new(Vector).Init(v.length, 0.0)
	for i := 0; i < len(v); i++ {
		outVec[i] = v[i] * vec[i]
	}
	return outVec
}

func (v *Vector) Scale(val float64) *Vector {
	outVec := new(Vector).Init(v.length, 0.0)
	for i := 0; i < len(v); i++ {
		outVec[i] = v[i] * val
	}
	return outVec
}

func (v *Vector) Sub(vec Vector) *Vector {
	outVec := new(Vector).Init(v.length, 0.0)
	for i := 0; i < len(v); i++ {
		outVec[i] = v[i] - vec[i]
	}
	return outVec
}

func (v *Vector) Sum() float64 {
	sum := 0
	for i := 0; i < len(v); i++ {
		sum += v[i]
	}
	return sum
}