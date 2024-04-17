package chp4

import (
	"fmt"
	"gorgonia.org/tensor"
)

type ActivationReLU struct {
	Output tensor.Tensor
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
}

type ActivationSoftMax struct {
	Output tensor.Tensor
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
