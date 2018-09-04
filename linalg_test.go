package gonet

import ("fmt")

func ExampleSimpleMatrixOperations() {
	m1 := new(Matrix).Init(2, 2)
	m1[0][0] = 1.0
	m1[0][1] = 2.0
	m1[1][0] = 3.0
	m1[1][1] = 4.0

	m2 := new(Matrix).Init(2, 2)
	m2[0][0] = 1.0
	m2[0][1] = 1.0
	m2[1][0] = 2.0
	m2[1][1] = 2.0

	v1 := new(Vector).Init(2, 1.0)

	fmt.Println(m1)
	fmt.Println(m2)
	fmt.Println(m1.Add(m2))
	
	fmt.Println(m1)
	fmt.Println(m2)
	fmt.Println(m1.Sub(m2))

	fmt.Println(m1)
	fmt.Println(m2)
	fmt.Println(m1.Scale(0.5))
	fmt.Println(m2.Scale(0.5))

	fmt.Println(m1)
	fmt.Println(m2)
	fmt.Println(v1)
	fmt.Println(m1.Apply(v1))
	fmt.Println(m1.ReverseApply(v1))
	fmt.Println(v1)

	// Output:
	// [[1 2] [3 4]]
	// [[1 1] [2 2]]
	// [[2 3] [5 6]]
	// [[1 2] [3 4]]
	// [[1 1] [2 2]]
	// [[0 1] [1 2]]
	// [[1 2] [3 4]]
	// [[1 1] [2 2]]
	// [[0.5 1] [1.5 2]]
	// [[0.5 0.5] [1 1]]
	// [[1 2] [3 4]]
	// [[1 1] [2 2]]
	// [1 1]
	// [3 7]
	// [4 6]
	// [1 1]
}

func ExampleSimpleVectorOperations() {
	v1 := new(Vector).Init(3, 1.0)
	v2 := new(Vector).Init(3, 2.0)
	v3 := new(Vector).Init(3, 1.0)
	v3[1] = 2.0
	v3[2] = 3.0
	v4 := new(Vector).Init(2, 2.0)

	fmt.Println(v1)
	fmt.Println(v2)
	fmt.Println(v3)
	fmt.Println(v4)
	fmt.Println(v3.Sum())
	fmt.Println(v1.Add(v2))
	fmt.Println(v1.Sub(v3))
	fmt.Println(v2.Mult(v3))
	fmt.Println(v1.Cross(v2))
	fmt.Println(v2.Cross(v3))
	fmt.Println(v2.Scale(0.5))
	fmt.Println(v4.Cross(v3).Apply(v2))
	fmt.Println(v3.Cross(v4).ReverseApply(v2))

	// Output:
	// [1 1 1]
	// [2 2 2]
	// [1 2 3]
	// [2 2]
	// 6
	// [3 3 3]
	// [0 -1 -2]
	// [2 4 6]
	// [[2 2 2] [2 2 2] [2 2 2]]
	// [[2 4 6] [2 4 6] [2 4 6]]
	// [1 1 1]
	// [24 24]
	// [24 24]
}