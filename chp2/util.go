package chp2

import t "gorgonia.org/tensor"

func AddBiases(inputs t.Tensor, biases []float64) (t.Tensor, error) {

	if len(inputs.Shape()) == 1 {
		_ = inputs.Reshape(inputs.Shape()[0], 1)
	}
	newShape := inputs.Shape()

	nB := make([]float64, newShape[0]*newShape[1])
	for i := range nB {
		nB[i] = biases[i%len(biases)]
	}
	newBiases := t.New(t.WithShape(newShape...), t.WithBacking(nB))
	return t.Add(inputs, newBiases)
}
