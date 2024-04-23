package chp7

import (
	"fmt"
	"gorgonia.org/tensor"
	"time"
)

func RunImpFunc1() {

	defer func(timer time.Time) {

		fmt.Println("  chapter 7  Impact : ", time.Since(timer))
	}(time.Now())
	p2Delta := 0.0001
	x1 := 3.0
	x2 := x1 + p2Delta
	y1 := f(x1)
	y2 := f(x2)
	approximateDerivative := (y2 - y1) / (x2 - x1)
	fmt.Println(approximateDerivative)
	//x := arange(0, 5, 0.001)
	//fmt.Println(x)
	b := y2 - (approximateDerivative * x2)
	fmt.Println(b)

}
func RunImpFunc2() {

	defer func(timer time.Time) {

		fmt.Println("  chapter 7  Impact : ", time.Since(timer))
	}(time.Now())
	//fmt.Println(y, x)
	x := []float64{0, 1, 2, 3, 4}
	y := make([]float64, len(x))

	for i := range x {
		y[i] = f(x[i])
	}

	fmt.Println(x)
	fmt.Println(y)

	// Calculate slope for the first pair of points
	slope1 := float64(y[1]-y[0]) / float64(x[1]-x[0])
	fmt.Println("Slope for the first pair of points:", slope1)

	// Calculate slope for the second pair of points
	slope2 := float64(y[3]-y[2]) / float64(x[3]-x[2])
	fmt.Println("Slope for the second pair of points:", slope2)
}

func f(x float64) float64 {
	return 2 * (x * x)
}

func arange(start, stop, step float64) *tensor.Dense {
	numElements := int((stop-start)/step) + 1

	x := tensor.New(tensor.WithShape(numElements), tensor.Of(tensor.Float64), tensor.WithBacking(make([]float64, numElements)))

	for i := 0; i < numElements; i++ {
		x.SetAt(start+float64(i)*step, i)
	}

	return x
}
