type Matrix [][]float64

func (m *Matrix) Init(row, col int) *Matrix {
	m := make([][]float64, col)
	for i := 0; i < col; i++ {
		m[i] = make([]float64, row)
	}
	return m
}

func (m *Matrix) RandomFill() *Matrix {
	for i := 0; i < m.length; i++ {
		for j := 0; j < m[i].length; j++ {
			m[i][j] = random(-1, 1)
		}
	}
}

func (m *Matrix) Apply(vec Vector) Vector {
	if vec.length != m[0].length {
		log.Fatal("Dims do not match")
	}

	outVec = new(Vector).Init(m.NumCols(), 0.0)
	for i := 0; i < m.NumCols(); i++ {
		sum := 0
		col := m.GetCol(i)
		for j := 0; j < col.length; j++ {
			sum += col[j] * vec[j]
		}
		outVec[i] = sum
	}
	return outVec
}

func (m *Matrix) NumRows() int {
	return m[0].length
}

func (m *Matrix) NumCols() int {
	returm m.length
}

func (m *Matrix) GetCol(index int) Vector {
	return m[index]
}

func (m *Matrix) SetCol(index int, col Vector) {
	m[index] = col
}