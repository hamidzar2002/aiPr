package chp3

import (
	"aiPr/chp2"
	"fmt"
	t "gorgonia.org/tensor"
)

type LayerDense struct {
	Weights             t.Tensor
	Biases              []float64
	Output              t.Tensor
	Input               t.Tensor
	DInput              t.Tensor
	DWeights            t.Tensor
	DBiases             []float64
	BiasMomentums       []float64
	WeightMomentums     t.Tensor
	BiasCache           []float64
	WeightCache         t.Tensor
	WeightRegularizerL1 float64
	WeightRegularizerL2 float64
	BiasRegularizerL1   float64
	BiasRegularizerL2   float64
}

func NewLayerDense(nInputs int, nNeurons int, Regulizer ...float64) LayerDense {

	weights := t.New(
		t.WithShape(nInputs, nNeurons),
		t.WithBacking(t.Random(t.Float64, nInputs*nNeurons)),
	)
	weights, err := weights.MulScalar(0.01, false)
	handleErr(err)

	biases := make([]float64, nNeurons)

	//set Regularizer
	var weightRegularizerL1, weightRegularizerL2, biasRegularizerL1, biasRegularizerL2 float64
	if len(Regulizer) == 0 {
		weightRegularizerL1, weightRegularizerL2, biasRegularizerL1, biasRegularizerL2 = 0, 0, 0, 0
	} else if len(Regulizer) == 4 {
		weightRegularizerL1, weightRegularizerL2, biasRegularizerL1, biasRegularizerL2 = Regulizer[0], Regulizer[1], Regulizer[2], Regulizer[3]
	} else {
		panic("not enough Regularizers has been set, they must be 4 ")
	}

	return LayerDense{
		Weights:             weights,
		Biases:              biases,
		WeightRegularizerL1: weightRegularizerL1,
		WeightRegularizerL2: weightRegularizerL2,
		BiasRegularizerL1:   biasRegularizerL1,
		BiasRegularizerL2:   biasRegularizerL2,
	}

}

func (l *LayerDense) Forward(inputs t.Tensor) {
	dp, err := t.Dot(inputs, l.Weights)
	handleErr(err)
	l.Output, err = chp2.AddBiases(dp, l.Biases)
	handleErr(err)
	l.Input = inputs.Clone().(t.Tensor)
}

func (l *LayerDense) Backward(dvalues t.Tensor) {
	var err error
	l.Input.T()
	l.DWeights, err = t.Dot(l.Input, dvalues)
	l.Input.T()
	handleErr(err)

	dbiases, err := t.Sum(dvalues, 0)
	handleErr(err)
	l.DBiases = dbiases.Data().([]float64)

	//add regularizer here logic here
	if l.WeightRegularizerL1 > 0 {
		dl := applyOneLikeAndNegetiveOneTensor(l.Weights)
		tmp, err := dl.MulScalar(l.WeightRegularizerL1, false)
		handleErr(err)
		l.DWeights, err = t.Add(l.DWeights, tmp)
		handleErr(err)
	}
	if l.WeightRegularizerL2 > 0 {
		w := t.New(t.WithShape(l.Weights.Shape()...), t.WithBacking(l.Weights.Data().([]float64)))
		tmp, err := w.MulScalar(2*l.WeightRegularizerL2, false)
		handleErr(err)
		l.DWeights, err = t.Add(l.DWeights, tmp)
		handleErr(err)
	}
	if l.BiasRegularizerL1 > 0 {
		dl := applyOneLikeAndNegetiveOneArray(l.Biases)
		for i := range dl {
			l.DBiases[i] = l.DBiases[i] + (l.BiasRegularizerL1 * dl[i])
		}
	}
	if l.BiasRegularizerL2 > 0 {

		for i := range l.DBiases {
			l.DBiases[i] = l.DBiases[i] + (l.BiasRegularizerL2 * 2 * l.Biases[i])
		}

	}

	//
	l.Weights.T()
	l.DInput, err = t.Dot(dvalues, l.Weights)
	l.Weights.T()
	handleErr(err)
}

func applyOneLikeAndNegetiveOneTensor(weights t.Tensor) *t.Dense {
	dL1 := make([]float64, len(weights.Data().([]float64)))
	for i := range weights.Data().([]float64) {

		if dL1[i] < 0 {
			dL1[i] = -1
		} else {
			dL1[i] = 1
		}
	}

	output := t.New(t.WithShape(weights.Shape()...), t.WithBacking(dL1))

	return output
}
func applyOneLikeAndNegetiveOneArray(biases []float64) []float64 {
	dL1 := make([]float64, len(biases))
	for i := range biases {
		if dL1[i] < 0 {
			dL1[i] = -1
		} else {
			dL1[i] = 1
		}
	}
	return dL1
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
