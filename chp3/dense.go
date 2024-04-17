package chp3

import (
	"aiPr/chp2"
	"fmt"
	t "gorgonia.org/tensor"
)

type LayerDense struct {
	Weights t.Tensor
	Biases  []float64
	Output  t.Tensor
}

func NewLayerDense(nInputs, nNeurons int) LayerDense {
	weights := t.New(
		t.WithShape(nInputs, nNeurons),
		t.WithBacking(t.Random(t.Float64, nInputs*nNeurons)),
	)
	weights, err := weights.MulScalar(0.01, false)
	handleErr(err)

	biases := make([]float64, nNeurons)

	return LayerDense{
		Weights: weights,
		Biases:  biases,
	}
}

func (l *LayerDense) Forward(inputs t.Tensor) {
	dp, err := t.Dot(inputs, l.Weights)
	handleErr(err)
	l.Output, err = chp2.AddBiases(dp, l.Biases)
	handleErr(err)
}
func PrintOutput(startI, amount int, data t.Tensor) {
	fmt.Print("[")
	for i := startI; i < amount; i++ {
		for j := 0; j < data.Shape()[1]; j++ {
			fmt.Print("[")
			s, err := data.At(i, j)
			handleErr(err)
			fmt.Print(s)
			//fmt.Print(",")
			fmt.Print("]")
		}
		fmt.Println(",")

	}
	fmt.Println("]")
}
