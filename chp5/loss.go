package chp5

import (
	"gorgonia.org/tensor"
	"math"
)

type Loss struct{}

// Calculate method calculates the data and regularization losses
// given model output and ground truth values
func (l *Loss) Calculate(output tensor.Tensor, y tensor.Tensor) float64 {
	// Calculate sample losses
	sampleLosses := l.forward(output, y)

	// Calculate mean loss
	//fmt.Println(sampleLosses[0])
	dataLoss := mean(sampleLosses)
	// Return loss
	return dataLoss
}

// forward method calculates sample losses
func (l *Loss) forward(yPred tensor.Tensor, yTrue tensor.Tensor) []float64 {

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

// mean calculates the mean of a slice of float64
func mean(data []float64) float64 {
	sum := 0.0
	for _, val := range data {
		sum += val
	}
	return sum / float64(len(data))
}
