type Vector []float64

func (v *Vector) Init(length int, initialFill float64) *Vector {
	v := make([]float64, length)
	for i := 0; i < length; i++ {
		v[i] = fill
	}
	return v
}

func (v *Vector) Mult(vec Vector) {
	outVec := new(Vector).Init(v.length, 0.0)
	for i := 0; i < v.length; i++ {
		outVec[i] = v[i] * vec[i]
	}
	return outVec
}

func (v *Vector) Sub(vec Vector) {
	outVec := new(Vector).Init(v.length, 0.0)
	for i := 0; i < v.length; i++ {
		outVec[i] = v[i] - vec[i]
	}
	return outVec
}