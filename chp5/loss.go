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
	} else if yTrue.Dims() == 2 {

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

	return negativeLogLikelihoods
}

// mean calculates the mean of a slice of float64
func mean(data []float64) float64 {
	sum := 0.0
	for _, val := range data {
		sum += val
	}
	return sum / float64(len(data))
}
