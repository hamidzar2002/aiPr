package chp4

import (
	"fmt"
	"gorgonia.org/tensor"
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
	r.Output = output
	r.Input = input
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
	r.DInput = drelu
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
	s.Output = output
}
func (s *ActivationSoftMax) Backward(input tensor.Tensor, grad tensor.Tensor, axis int) {

	input, err := tensor.SoftMaxB(input, grad, axis)
	if err != nil {
		fmt.Println("error", err)
		return
	}
	s.DInput = input
}
