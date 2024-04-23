package chp9

import (
	"aiPr/chp2"
	"aiPr/chp3"
	"aiPr/chp4"
	"fmt"
	"gorgonia.org/tensor"
)

func RunBackpropagationMultiFunc1() {

	dvaluesB := []float64{1.0, 1.0, 1.0}
	dvalues := tensor.New(tensor.WithShape(1, 3), tensor.WithBacking(dvaluesB))

	weightsB := []float64{0.2, 0.8, -0.5, 1,
		0.5, -0.91, 0.26, -0.5,
		-0.26, -0.27, 0.17, 0.87}
	weights := tensor.New(tensor.WithShape(3, 4), tensor.WithBacking(weightsB))
	//weights.T()
	x, _ := weights.Sum(0)
	var dinputs []float64
	for i, val := range dvalues.Data().([]float64) {
		if i == 0 {
			for _, xx := range x.Data().([]float64) {
				dinput := xx * val
				dinputs = append(dinputs, dinput)
			}
		}

	}
	fmt.Println("dinputs1", dinputs)
	//weights.T()
	dinputs2, _ := tensor.Dot(dvalues, weights)
	fmt.Println("dinputs2", dinputs2)
	//y, err := x.Mul(dvalues)
	//fmt.Println(dvalues, weights, x, y, err)

	return
}
func RunBackpropagationMultiFunc2() {

	dvaluesB := []float64{1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0}
	dvalues := tensor.New(tensor.WithShape(3, 3), tensor.WithBacking(dvaluesB))

	weightsB := []float64{0.2, 0.8, -0.5, 1,
		0.5, -0.91, 0.26, -0.5,
		-0.26, -0.27, 0.17, 0.87}
	weights := tensor.New(tensor.WithShape(3, 4), tensor.WithBacking(weightsB))
	weights.T()
	weights.T()
	dinputs, _ := tensor.Dot(dvalues, weights)
	fmt.Println("dinputs", dinputs)
	//y, err := x.Mul(dvalues)
	//fmt.Println(dvalues, weights, x, y, err)

	return
}
func RunBackpropagationMultiFunc3() {

	dvaluesB := []float64{1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0}
	dvalues := tensor.New(tensor.WithShape(3, 3), tensor.WithBacking(dvaluesB))

	inputsB := []float64{1, 2, 3, 2.5,
		2., 5., -1., 2,
		-1.5, 2.7, 3.3, -0.8}
	inputs := tensor.New(tensor.WithShape(3, 4), tensor.WithBacking(inputsB))
	inputs.T()
	//weights.T()
	dweights, _ := tensor.Dot(inputs, dvalues)
	fmt.Println("dweights", dweights)
	//y, err := x.Mul(dvalues)
	//fmt.Println(dvalues, weights, x, y, err)

	return
}
func RunBackpropagationBiasFunc4() {

	dvaluesB := []float64{1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0}
	dvalues := tensor.New(tensor.WithShape(3, 3), tensor.WithBacking(dvaluesB))

	biasB := []float64{2, 3, 0.5}
	biases := tensor.New(tensor.WithShape(1, 3), tensor.WithBacking(biasB))
	fmt.Println("biases", biases)

	dbiases, _ := tensor.Sum(dvalues, 0)
	fmt.Println("dbiases", dbiases)

	return
}

func RunBackpropagationReLUFunc5() {

	dvaluesB := []float64{1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12}
	dvalues := tensor.New(tensor.WithShape(3, 4), tensor.WithBacking(dvaluesB))

	zB := []float64{1, 2, -3, -4,
		2, -7, -1, 3,
		-1, 2, 5, -1}
	z := tensor.New(tensor.WithShape(3, 4), tensor.WithBacking(zB))

	var dreluB []float64
	for _, val := range z.Data().([]float64) {
		if val > 0 {
			dreluB = append(dreluB, 1)
		} else {
			dreluB = append(dreluB, 0)
		}

	}

	drelu := tensor.New(tensor.WithShape(z.Shape()...), tensor.WithBacking(dreluB))
	//fmt.Println("drelu", drelu)

	drelu, _ = drelu.Mul(dvalues)
	fmt.Println("drelu after mul", drelu)

	//second way :
	it := z.Iterator()
	drelu2 := dvalues.Clone().(tensor.Tensor)
	for _, err := it.Start(); err == nil; _, err = it.Next() {
		val, _ := z.At(it.Coord()...)
		if val.(float64) <= 0 {
			drelu2.SetAt(0.0, it.Coord()...)
		}
	}

	fmt.Println("second way:", drelu2)

	return
}

func RunBackpropagationAllFunc6() {

	dvaluesB := []float64{1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0}
	inputsB := []float64{1, 2, 3, 2.5, 2., 5., -1., 2, -1.5, 2.7, 3.3, -0.8}
	biasB := []float64{2, 3, 0.5}
	weightsB := []float64{0.2, 0.8, -0.5, 1, 0.5, -0.91, 0.26, -0.5, -0.26, -0.27, 0.17, 0.87}
	dvalues := tensor.New(tensor.WithShape(3, 3), tensor.WithBacking(dvaluesB))
	inputs := tensor.New(tensor.WithShape(3, 4), tensor.WithBacking(inputsB))
	biases := tensor.New(tensor.WithShape(1, 3), tensor.WithBacking(biasB))
	weights := tensor.New(tensor.WithShape(3, 4), tensor.WithBacking(weightsB))
	weights.T()

	fmt.Println("dvalues", dvalues)
	//forward
	dp, _ := tensor.Dot(inputs, weights)
	layer_outputs, _ := chp2.AddBiases(dp, biases.Data().([]float64))
	fmt.Println("layer_outputs", layer_outputs)
	//activation
	zeros := tensor.New(tensor.WithShape(layer_outputs.Shape()...), tensor.Of(tensor.Float64))
	relu_outputs, _ := tensor.MaxBetween(layer_outputs, zeros)

	//second way :
	it := inputs.Iterator()
	drelu := relu_outputs.Clone().(tensor.Tensor)
	for _, err := it.Start(); err == nil; _, err = it.Next() {
		val, _ := inputs.At(it.Coord()...)
		if val.(float64) <= 0 {
			drelu.SetAt(0.0, it.Coord()...)
		}
	}

	weights.T()
	dinputs, _ := tensor.Dot(drelu, weights)
	fmt.Println(dinputs)
	weights.T()

	inputs.T()
	dweights, _ := tensor.Dot(inputs, drelu)
	inputs.T()
	dbiases, _ := tensor.Sum(drelu, 0)

	//fmt.Println("final dinputs:", dinputs)
	//fmt.Println("final dweights:", dweights)
	//fmt.Println("final dbiases:", dbiases)

	//new weights::
	dweightsTemp := tensor.New(tensor.WithShape(dweights.Shape()...), tensor.WithBacking(dweights.Data()))
	deweightsS, _ := dweightsTemp.MulScalar(-0.001, false)
	//weights.T()
	weights, _ = weights.Add(deweightsS)
	fmt.Println("New weights:", weights)

	//new biases
	dbiasesTmp := tensor.New(tensor.WithShape(dbiases.Shape()...), tensor.WithBacking(dbiases.Data()))
	dbiasesM, _ := dbiasesTmp.MulScalar(-0.001, false)
	biases, _ = biases.Add(dbiasesM)
	fmt.Println("New biases", biases)

	return
}

func RunBackpropagationCalcFunc7() {
	inputsB := []float64{1, 2, 3, 2.5, 2., 5., -1., 2, -1.5, 2.7, 3.3, -0.8}
	biasB := []float64{2, 3, 0.5}
	weightsB := []float64{0.2, 0.8, -0.5, 1, 0.5, -0.91, 0.26, -0.5, -0.26, -0.27, 0.17, 0.87}
	inputs := tensor.New(tensor.WithShape(3, 4), tensor.WithBacking(inputsB))
	biases := tensor.New(tensor.WithShape(1, 3), tensor.WithBacking(biasB))
	weights := tensor.New(tensor.WithShape(3, 4), tensor.WithBacking(weightsB))

	//define layers
	dens := chp3.LayerDense{
		Input:   inputs,
		Weights: weights,
		Biases:  biasB,
	}
	act := chp4.NewActivationReLU()
	//start forwarding
	dens.Weights.T()
	dens.Forward(inputs)
	act.Forward(dens.Output)

	//Now we should backward step
	act.Backward(act.Output)
	dens.Backward(act.DInput)

	//now calculate new wieths and biases to reduce loss
	dweightsTemp := tensor.New(tensor.WithShape(dens.DWeights.Shape()...), tensor.WithBacking(dens.DWeights.Data()))
	deweightsS, _ := dweightsTemp.MulScalar(-0.001, false)
	weights, _ = weights.Add(deweightsS)
	fmt.Println("New weights:", weights)

	//new biases
	dbiasesTmp := tensor.New(tensor.WithShape(biases.Shape()...), tensor.WithBacking(dens.DBiases))
	dbiasesM, _ := dbiasesTmp.MulScalar(-0.001, false)
	biases, _ = biases.Add(dbiasesM)
	fmt.Println("New biases", biases)

	return
}
