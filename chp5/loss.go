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
	sampleLosses := l.forward2(output, y)

	// Calculate mean loss
	//fmt.Println(sampleLosses[0])
	dataLoss := mean(sampleLosses)
	// Return loss
	return dataLoss
}

// forward method calculates sample losses
func (l *Loss) forward(yPred tensor.Tensor, yTrue tensor.Tensor) []float64 {
	// Number of samples in a batch
	samples := yPred.Shape()[0]

	// Clip data to prevent division by 0
	yPredClipped := yPred.Clone().(tensor.Tensor)

	yPredClipped.Apply(func(e float64) float64 {
		return math.Max(1e-7, math.Min(e, 1-1e-7))
	})

	// Probabilities for target values - only if categorical labels
	var correctConfidences []float64
	if yTrue.Dims() == 1 || yTrue.Shape()[1] == 1 {

		correctConfidences = make([]float64, samples)
		yTrueData := yTrue.Data().([]float64)
		yPredData := yPredClipped.Data().([]float64)
		for i := 0; i < samples; i++ {
			idx := i * yTrue.Strides()[0]

			val := yPredData[int(yTrueData[idx])]
			correctConfidences[i] = val
		}
		//fmt.Println(correctConfidences)
	} else {

		correctConfidences = make([]float64, samples)
		yTrueData := yTrue.Data().([]float64)
		yPredData := yPredClipped.Data().([]float64)
		for i := 0; i < samples; i++ {
			sum := 0.0
			for j := 0; j < yTrue.Shape()[1]; j++ {
				sum += yPredData[i*yTrue.Strides()[0]+j] * yTrueData[i*yTrue.Strides()[0]+j]
			}

			correctConfidences[i] = sum
		}
	}

	// Losses
	negativeLogLikelihoods := make([]float64, samples)
	for i, val := range correctConfidences {
		negativeLogLikelihoods[i] = -math.Log(val)
	}

	//return correctConfidences
	return negativeLogLikelihoods
}

func (l *Loss) forward2(yPred tensor.Tensor, yTrue tensor.Tensor) []float64 {

	it := yPred.Iterator()

	predictedValsBacking := make([]float64, yPred.Shape()[0])
	//if yTrue.Dims() == 2 {
	x := yTrue.Data().([]float64)
	for _, err := it.Start(); err == nil; _, err = it.Next() {
		//val, _ := softmaxOutput.At(it.Coord()...)

		c := it.Coord()
		//correctIndex, _ := yTrue.At(c[0])
		correctIndex := x[c[0]]
		//fmt.Println(it.Coord())
		//fmt.Println(correctIndex, c[0], c[1])

		if correctIndex == float64(c[1]) {
			//	predictedValsBacking[]
			//fmt.Println("wwww", correctIndex, c[0], c[1])
			tval, _ := yPred.At(c...)
			val := tval.(float64)
			predictedValsBacking[c[0]] = val
		}
	}
	//} else if yTrue.Dims() == 2 {
	//
	//	argMax, _ := tensor.Argmax(yTrue, 1)
	//	//val, _ := softmaxOutput.At(it.Coord()...)
	//	//fmt.Println(i, it.Coord(), val)
	//	for _, err := it.Start(); err == nil; _, err = it.Next() {
	//		c := it.Coord()
	//
	//		correctIndex, _ := argMax.At(c[0])
	//		if correctIndex == int(c[1]) {
	//
	//			//	predictedValsBacking[]
	//			tval, _ := yPred.At(c...)
	//			val := tval.(float64)
	//
	//			yPred.Data().([]float64)[c[0]] = val
	//		}
	//	}
	//}
	predictedVals := tensor.New(tensor.WithBacking(predictedValsBacking))

	losses, _ := tensor.Log(predictedVals)
	//fmt.Println("wwww", predictedVals.Shape(), losses.Shape())

	losses, _ = tensor.Mul(losses, float64(-1))
	//fmt.Println(losses)
	return losses.Data().([]float64)
	//// Losses
	//negativeLogLikelihoods := make([]float64, samples)
	//for i, val := range correctConfidences {
	//	negativeLogLikelihoods[i] = -math.Log(val)
	//}
	//
	//return negativeLogLikelihoods
}

// mean calculates the mean of a slice of float64
func mean(data []float64) float64 {
	sum := 0.0
	for _, val := range data {
		sum += val
	}
	return sum / float64(len(data))
}
