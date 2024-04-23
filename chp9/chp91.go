package chp9

import (
	"fmt"
	"gorgonia.org/tensor"
	"math"
)

func RunBackpropagationFunc1() {

	// Input values
	xs := []float64{1.0, -2.0, 3.0}
	x := tensor.New(tensor.WithShape(1, 3), tensor.Of(tensor.Float64), tensor.WithBacking(xs))
	// Weights
	ws := []float64{-3.0, -1.0, 2.0}
	w := tensor.New(tensor.WithShape(1, 3), tensor.Of(tensor.Float64), tensor.WithBacking(ws))
	// Bias
	b := 1.0
	fmt.Println(x, w, b)
	x0, _ := x.At(0, 0)
	w0, _ := w.At(0, 0)
	x1, _ := x.At(0, 1)
	w1, _ := w.At(0, 1)
	x2, _ := x.At(0, 2)
	w2, _ := w.At(0, 2)
	xw0 := x0.(float64) * w0.(float64)
	xw1 := x1.(float64) * w1.(float64)
	xw2 := x2.(float64) * w2.(float64)
	fmt.Println("xws:", xw0, xw1, xw2)
	z := xw0 + xw1 + xw2 + b
	fmt.Println("sum", z)
	y := math.Max(z, 0)
	fmt.Println("ReLU", y)

	//derivation
	dvalue := float64(1)
	drelu_dz := dvalue * DerivativeOfReLU(z)
	fmt.Println("drelu_dz", drelu_dz)
	dsumxw0 := float64(1)
	dsumxw1 := float64(1)
	dsumxw2 := float64(1)
	dsum_db := float64(1)
	drelu_dxw0 := drelu_dz * dsumxw0
	drelu_dxw1 := drelu_dz * dsumxw1
	drelu_dxw2 := drelu_dz * dsumxw2
	drelu_db := drelu_dz * dsum_db
	fmt.Println("drelu_dxws", drelu_dxw0, drelu_dxw1, drelu_dxw2)
	fmt.Println("drelu_db", drelu_db)

	dmul_dx0 := w0.(float64)
	dmul_dx1 := w1.(float64)
	dmul_dx2 := w2.(float64)
	dmul_dw0 := x0.(float64)
	dmul_dw1 := x1.(float64)
	dmul_dw2 := x2.(float64)

	drelu_dx0 := drelu_dxw0 * dmul_dx0
	drelu_dw0 := drelu_dxw0 * dmul_dw0
	drelu_dx1 := drelu_dxw1 * dmul_dx1
	drelu_dw1 := drelu_dxw1 * dmul_dw1
	drelu_dx2 := drelu_dxw2 * dmul_dx2
	drelu_dw2 := drelu_dxw2 * dmul_dw2
	fmt.Println("drelu_dxs", drelu_dx0, drelu_dx1, drelu_dx2)
	fmt.Println("drelu_dws", drelu_dw0, drelu_dw1, drelu_dw2)

	//final formula of derivative of dxs amd dws
	drelu_dx0 = dvalue * DerivativeOfReLU(y) * w0.(float64)
	drelu_dx1 = dvalue * DerivativeOfReLU(y) * w1.(float64)
	drelu_dx2 = dvalue * DerivativeOfReLU(y) * w2.(float64)
	drelu_dw0 = dvalue * DerivativeOfReLU(y) * x0.(float64)
	drelu_dw1 = dvalue * DerivativeOfReLU(y) * x1.(float64)
	drelu_dw2 = dvalue * DerivativeOfReLU(y) * x2.(float64)
	//fmt.Println(drelu_dx0)
	dxs := []float64{drelu_dx0, drelu_dx1, drelu_dx2}
	dws := []float64{drelu_dw0, drelu_dw1, drelu_dw2}
	db := drelu_db

	fmt.Println("final", dxs, dws, db)

	// change weights based on dws
	ws[0] += -0.001 * dws[0]
	ws[1] += -0.001 * dws[1]
	ws[2] += -0.001 * dws[2]
	b += -0.001 * db

	fmt.Println("new ws, b, xs", ws, b)

	//re-run the network
	x = tensor.New(tensor.WithShape(1, 3), tensor.Of(tensor.Float64), tensor.WithBacking(xs))
	w = tensor.New(tensor.WithShape(1, 3), tensor.Of(tensor.Float64), tensor.WithBacking(ws))

	x0, _ = x.At(0, 0)
	w0, _ = w.At(0, 0)
	x1, _ = x.At(0, 1)
	w1, _ = w.At(0, 1)
	x2, _ = x.At(0, 2)
	w2, _ = w.At(0, 2)
	xw0 = x0.(float64) * w0.(float64)
	xw1 = x1.(float64) * w1.(float64)
	xw2 = x2.(float64) * w2.(float64)
	fmt.Println("xws:", xw0, xw1, xw2)
	z = xw0 + xw1 + xw2 + b
	fmt.Println("sum", z)
	y = math.Max(z, 0)
	fmt.Println("ReLU", y)

	return
}

func DerivativeOfReLU(num float64) float64 {
	if num <= 0 {
		return 0.0
	}
	return 1.0
}
