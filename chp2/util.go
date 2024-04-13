package chp2

import t "gorgonia.org/tensor"

func AddBiases(inputs t.Tensor, biases []float32) (t.Tensor, error) {
	newShape := inputs.Shape()
	nB := make([]float32, newShape[0]*newShape[1])
	for i := range nB {
		nB[i] = biases[i%len(biases)]
	}
	newBiases := t.New(t.WithShape(newShape...), t.WithBacking(nB))
	return t.Add(inputs, newBiases)
}
