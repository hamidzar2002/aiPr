package chp15

import (
	"gonum.org/v1/gonum/stat/distuv"
	"gorgonia.org/tensor"
)

type Dropout struct {
	Rate       float64
	BinaryMask []float64
	Inputs     tensor.Tensor
	Output     tensor.Tensor
	DInputs    tensor.Tensor
}

func NewDropout(rate float64) Dropout {
	return Dropout{Rate: 1 - rate}
}

func (d *Dropout) Forward(input tensor.Tensor) {
	d.Inputs = input
	d.BinaryMask = binomial(1, d.Rate, input.Shape()[0]*input.Shape()[1])
	for i := range d.BinaryMask {
		d.BinaryMask[i] = d.BinaryMask[i] / d.Rate
	}
	inpBack := input.Data().([]float64)
	for i := range inpBack {
		inpBack[i] = inpBack[i] * d.BinaryMask[i]
	}
	d.Output = tensor.New(tensor.WithShape(input.Shape()...), tensor.WithBacking(inpBack))
}
func (d *Dropout) Backward(dValues tensor.Tensor) {
	dVBack := dValues.Data().([]float64)
	for i := range dVBack {
		dVBack[i] = dVBack[i] * d.BinaryMask[i]
	}
	d.DInputs = tensor.New(tensor.WithShape(dValues.Shape()...), tensor.WithBacking(dVBack))
}

func binomial(n float64, p float64, size int) []float64 {
	// Create a binomial distribution
	bin := distuv.Binomial{
		N: n,
		P: p,
	}

	// Generate random samples
	samples := make([]float64, size)
	for i := range samples {
		samples[i] = bin.Rand()
	}

	return samples
}
