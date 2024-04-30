package chp5

import (
	"aiPr/chp4"
	"fmt"
	"gorgonia.org/tensor"
	"math"
)

type Loss struct {
	DInputs  tensor.Tensor
	DataLoss float64
}
type ActivationSoftmaxLoss struct {
	ActivationSoftMax chp4.ActivationSoftMax
	Loss              Loss
	DInputs           tensor.Tensor
}

// Calculate method calculates the data and regularization losses
// given model output and ground truth values
func (l *Loss) Calculate(output tensor.Tensor, y tensor.Tensor) float64 {
	// Calculate sample losses
	sampleLosses := l.Forward(output, y)

	// Calculate mean loss
	//fmt.Println(sampleLosses[0])
	dataLoss := mean(sampleLosses)
	// Return loss
	l.DataLoss = dataLoss
	return dataLoss
}

// forward method calculates sample losses
func (l *Loss) Forward(yPred tensor.Tensor, yTrue tensor.Tensor) []float64 {

	it := yPred.Iterator()

	predictedValsBacking := make([]float64, yPred.Shape()[0])
	yPred.Apply(func(e float64) float64 {
		return math.Max(1e-7, math.Min(e, 1-1e-7))
	})
	if yTrue.Dims() == 1 || yTrue.Shape()[1] == 1 {
		x := yTrue.Data().([]float64)
		for _, err := it.Start(); err == nil; _, err = it.Next() {
			c := it.Coord()
			correctIndex := x[c[0]]
			if correctIndex == float64(c[1]) {
				tval, _ := yPred.At(c...)
				val := tval.(float64)
				predictedValsBacking[c[0]] = val
			}
		}
	} else if yTrue.Dims() == 2 {
		argMax, _ := tensor.Argmax(yTrue, 1)
		for _, err := it.Start(); err == nil; _, err = it.Next() {
			c := it.Coord()
			tmp, _ := argMax.At(c[0])
			correctIndex := float64(tmp.(int))
			if correctIndex == float64(c[1]) {
				tval, _ := yPred.At(c...)
				val := tval.(float64)
				predictedValsBacking[c[0]] = val
			}
		}
	}

	predictedVals := tensor.New(tensor.WithBacking(predictedValsBacking))
	losses, _ := tensor.Log(predictedVals)
	losses, _ = tensor.Mul(losses, float64(-1))
	return losses.Data().([]float64)

}

func (l *Loss) Backward(dvalues tensor.Tensor, yTrue tensor.Tensor) {
	var err error
	// Number of samples
	samples := dvalues.Shape()[0]
	// Number of labels in every sample
	// We'll use the first sample to count them
	labels := dvalues.Shape()[1]

	//fmt.Println(yTrue)
	var dinputs tensor.Tensor
	if len(yTrue.Shape()) == 1 || yTrue.Shape()[1] == 1 {
		yTrueOneHot := Eye(labels, yTrue.Data().([]float64))
		yTrue = tensor.New(tensor.WithShape(dvalues.Shape()...), tensor.WithBacking(yTrueOneHot.Data().([]float64)))

	}
	// Calculate gradient
	dinputs, err = tensor.Div(yTrue, dvalues)
	handleErr(err)
	dinputs, err = tensor.Mul(dinputs, -1.0)
	handleErr(err)

	// Normalize gradient
	//samp := tensor.New(tensor.WithBacking(samples))
	dinputs, err = tensor.Div(dinputs, float64(samples))
	handleErr(err)

	l.DInputs = dinputs
	//return dinputs.Data().([]float64)
}

func NewActivationSoftmaxLoss() ActivationSoftmaxLoss {

	var loss Loss
	return ActivationSoftmaxLoss{ActivationSoftMax: chp4.NewActivationSoftMax(), Loss: loss}
}

func (ASL *ActivationSoftmaxLoss) Forward(input tensor.Tensor, yTrue tensor.Tensor) float64 {
	ASL.ActivationSoftMax.Forward(input, 1)
	return ASL.Loss.Calculate(ASL.ActivationSoftMax.Output, yTrue)

}

func (ASL *ActivationSoftmaxLoss) Backward(dvalues tensor.Tensor, yTrue tensor.Tensor) {

	var err error

	//input, err := tensor.SoftMaxB(dvalues, dvalues, 1)
	//handleErr(err)
	//ASL.DInputs = input

	// Number of samples
	samples := dvalues.Shape()[0]

	var dinputs tensor.Tensor
	var yyTrue *tensor.Dense
	if yTrue.Dims() == 2 && yTrue.Shape()[1] == 1 {
		yyTrue = tensor.New(tensor.WithBacking(yTrue.Data()))
	} else {
		yTrue, _ = tensor.Argmax(yTrue, 1)
		temp := make([]float64, len(yTrue.Data().([]int)))
		for val := range yTrue.Data().([]int) {
			temp = append(temp, float64(val))
		}
		yyTrue = tensor.New(tensor.WithBacking(temp))
	}

	dinputs = dvalues.Clone().(tensor.Tensor)
	for i := 0; i < samples; i++ {
		valy := yyTrue.Get(i)
		idx := int(valy.(float64))
		x, _ := dinputs.At(i, idx)
		dinputs.SetAt(x.(float64)-1.0, i, idx)

	}
	// Normalize gradient
	dinputs, err = tensor.Div(dinputs, float64(samples))

	handleErr(err)
	ASL.DInputs = dinputs
}

// mean calculates the mean of a slice of float64
func mean(data []float64) float64 {
	sum := 0.0
	for _, val := range data {
		sum += val
	}
	return sum / float64(len(data))
}

func Eye(n int, y []float64) *tensor.Dense {
	ey := tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(n, n))
	for i := 0; i < n; i++ {
		idxY := int(y[i])
		ey.SetAt(1.0, i, idxY)
	}
	return ey
}

func handleErr(er error) {
	if er != nil {
		fmt.Println("error", er)
	}
}

func Accuracy(predictions []int, y []float64) float64 {
	var correct int
	for i, pred := range predictions {
		if float64(pred) == y[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(y))
}
