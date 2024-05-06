package chp4

import (
	"fmt"
	"gorgonia.org/tensor"
	"math"
)

type ActivationReLU struct {
	Output tensor.Tensor
	Input  tensor.Tensor
	DInput tensor.Tensor
}

func NewActivationReLU() ActivationReLU {
	return ActivationReLU{Output: tensor.New(tensor.Of(tensor.Float64))}
}

func (r *ActivationReLU) Forward(input tensor.Tensor) {

	zeros := tensor.New(tensor.WithShape(input.Shape()...), tensor.Of(tensor.Float64))

	output, err := tensor.MaxBetween(input, zeros)
	if err != nil {
		fmt.Println("error", err)
		return
	}
	r.Output = output.Clone().(tensor.Tensor)
	r.Input = input.Clone().(tensor.Tensor)
}

func (r *ActivationReLU) Backward(dvalues tensor.Tensor) {

	var err error
	//second way :
	it := r.Input.Iterator()
	drelu := dvalues.Clone().(tensor.Tensor)
	for _, errr := it.Start(); errr == nil; _, errr = it.Next() {
		val, _ := r.Input.At(it.Coord()...)
		if val.(float64) <= 0 {
			drelu.SetAt(0.0, it.Coord()...)
		}
	}

	if err != nil {
		fmt.Println("error", err)
		return
	}
	r.DInput = drelu.Clone().(tensor.Tensor)
}

type ActivationSoftMax struct {
	Output tensor.Tensor
	DInput tensor.Tensor
}

func NewActivationSoftMax() ActivationSoftMax {
	return ActivationSoftMax{Output: tensor.New(tensor.Of(tensor.Float64))}
}

func (s *ActivationSoftMax) Forward(input tensor.Tensor, axis int) {

	output, err := tensor.SoftMax(input, axis)
	if err != nil {
		fmt.Println("error", err)
		return
	}
	s.Output = output.Clone().(tensor.Tensor)
}
func (s *ActivationSoftMax) Backward(input tensor.Tensor, grad tensor.Tensor, axis int) {

	input, err := tensor.SoftMaxB(input, grad, axis)
	if err != nil {
		fmt.Println("error", err)
		return
	}
	s.DInput = input.Clone().(tensor.Tensor)
}

type ActivationSigmoid struct {
	Outputs tensor.Tensor
	DInputs tensor.Tensor
	Inputs  tensor.Tensor
}

func NewActivationSigmoid() ActivationSigmoid {
	return ActivationSigmoid{Outputs: tensor.New(tensor.Of(tensor.Float64))}
}

func (as *ActivationSigmoid) Forward(input tensor.Tensor) {

	as.Inputs = input
	inputsBk := as.Inputs.Data().([]float64)
	OutputsBk := make([]float64, len(inputsBk))
	for i := range inputsBk {
		OutputsBk[i] = 1 / (1 + math.Exp(-inputsBk[i]))
	}

	as.Outputs = tensor.New(tensor.WithShape(as.Inputs.Shape()...), tensor.WithBacking(OutputsBk))
}
func (as *ActivationSigmoid) Backward(dValues tensor.Tensor) {

	as.DInputs = dValues
	OutputsBk := as.Outputs.Data().([]float64)
	dValuesBk := dValues.Data().([]float64)
	DInputsBk := as.DInputs.Data().([]float64)

	for i, val := range dValuesBk {
		DInputsBk[i] = val * (1 - OutputsBk[i]) * OutputsBk[i]
	}
	as.DInputs = tensor.New(tensor.WithShape(dValues.Shape()...), tensor.WithBacking(DInputsBk))
}

type ActivationLinear struct {
	Outputs tensor.Tensor
	DInputs tensor.Tensor
	Inputs  tensor.Tensor
}

func NewActivationLinear() ActivationLinear {
	return ActivationLinear{Outputs: tensor.New(tensor.Of(tensor.Float64))}
}

func (as *ActivationLinear) Forward(input tensor.Tensor) {
	as.Inputs = input
	as.Outputs = input
}
func (as *ActivationLinear) Backward(dValues tensor.Tensor) {
	as.DInputs = dValues.Clone().(tensor.Tensor)

}
