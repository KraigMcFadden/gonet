package gonet

import (
	"log"
	"strconv"
)

type Matrix [][]float64

func (m Matrix) Init(row, col int) Matrix {
	m = make(Matrix, row)
	for i := 0; i < row; i++ {
		m[i] = make([]float64, col)
	}
	return m
}

func (m Matrix) RandomFill() Matrix {
	for i := 0; i < m.NumRows(); i++ {
		for j := 0; j < m.NumCols(); j++ {
			m[i][j] = random(-1, 1)
		}
	}
	return m
}

func (m Matrix) Apply(vec Vector) Vector {
	if len(vec) != m.NumCols() {
		log.Fatal("Matrix.Apply(): Dims do not match! Vector: " + strconv.Itoa(len(vec)) + 
			" Mat Cols: " + strconv.Itoa(m.NumCols()))
	}

	outVec := new(Vector).Init(m.NumRows(), 0.0)
	for i := 0; i < m.NumRows(); i++ {
		row := m[i]
		for j := 0; j < m.NumCols(); j++ {
			outVec[i] += row[j] * vec[j]
		}
	}
	return outVec
}

func (m Matrix) ReverseApply(vec Vector) Vector {
	if len(vec) != m.NumRows() {
		log.Fatal("Matrix.ReverseApply(): Dims do not match! Vector: " + strconv.Itoa(len(vec)) + 
			" Mat Rows: " + strconv.Itoa(m.NumRows()))
	}

	outVec := new(Vector).Init(m.NumCols(), 0.0)
	for i := 0; i < m.NumRows(); i++ {
		for j := 0; j < m.NumCols(); j++ {
			outVec[j] += m[i][j] * vec[i]
		}
	}
	return outVec
}

func (m Matrix) Add(mat Matrix) Matrix {
	if m.NumRows() != mat.NumRows() || m.NumCols() != mat.NumCols() {
		log.Fatal("Matrix.Add(): Dims do not match! This rows: " + strconv.Itoa(m.NumRows()) + 
			" This cols: " + strconv.Itoa(m.NumCols()) + " That rows: " + strconv.Itoa(mat.NumRows()) + 
			" That cols: " + strconv.Itoa(mat.NumCols()))
	}

	outMat := new(Matrix).Init(m.NumRows(), m.NumCols())
	for i := 0; i < m.NumRows(); i++ {
		for j := 0; j < m.NumCols(); j++ {
			outMat[i][j] = m[i][j] + mat[i][j]
		}
	}
	return outMat
}

func (m Matrix) Scale(val float64) Matrix {
	outMat := new(Matrix).Init(m.NumRows(), m.NumCols())
	for i := 0; i < m.NumRows(); i++ {
		for j := 0; j < m.NumCols(); j++ {
			outMat[i][j] = m[i][j] * val
		}
	}
	return outMat
}

func (m Matrix) NumRows() int {
	return len(m)
}

func (m Matrix) NumCols() int {
	return len(m[0])
}

func (m Matrix) GetElement(row, col int) float64 {
	return m[row][col]
}

func (m Matrix) SetElement(row, col int, value float64) {
	m[row][col] = value
}