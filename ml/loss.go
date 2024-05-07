package ml

import (
	"gorgonia.org/tensor"
	"math"
)

type Los interface {
	//Calculate(output tensor.Tensor, y tensor.Tensor) float64
	Forward(yPred tensor.Tensor, yTrue tensor.Tensor) float64
	Backward(dvalues tensor.Tensor, yTrue tensor.Tensor)
	//RegularizationLoss(layer ml.LayerDense) float64
}

type NormalLoss struct {
	DInputs  tensor.Tensor
	DataLoss float64
}
type ActivationSoftmaxLoss struct {
	ActivationSoftMax *ActivationSoftMax
	Loss              NormalLoss
	DInputs           tensor.Tensor
}
type BinaryCrossentropyLoss struct {
	DInputs  tensor.Tensor
	DataLoss float64
	Loss     NormalLoss
}
type MeanSquaredErrorLoss struct {
	DInputs  tensor.Tensor
	DataLoss float64
	Loss     NormalLoss
}
type MeanAbsoluteErrorLoss struct {
	DInputs  tensor.Tensor
	DataLoss float64
	Loss     NormalLoss
}

// Calculate method calculates the data and regularization losses
// given model output and ground truth values
func (l *NormalLoss) Calculate(output tensor.Tensor, y tensor.Tensor) float64 {
	// Calculate sample losses
	sampleLosses := l.Forward(output, y)
	return sampleLosses
}

// forward method calculates sample losses
func (l *NormalLoss) Forward(yPred tensor.Tensor, yTrue tensor.Tensor) float64 {

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

	dataLoss := mean(losses.Data().([]float64))
	// Return loss
	l.DataLoss = dataLoss
	return dataLoss

}

func (l *NormalLoss) Backward(dvalues tensor.Tensor, yTrue tensor.Tensor) {
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

	l.DInputs = dinputs.Clone().(tensor.Tensor)
	//return dinputs.Data().([]float64)
}
func (l *NormalLoss) RegularizationLoss(layer *LayerDense) float64 {
	var regularizationLoss float64

	// L1 regularization - weights
	if layer.WeightRegularizerL1 > 0 {
		for _, weight := range layer.Weights.Data().([]float64) {
			regularizationLoss += layer.WeightRegularizerL1 * math.Abs(weight)
		}
	}

	// L2 regularization - weights
	if layer.WeightRegularizerL2 > 0 {
		for _, weight := range layer.Weights.Data().([]float64) {
			regularizationLoss += layer.WeightRegularizerL2 * weight * weight
		}
	}

	// L1 regularization - biases
	if layer.BiasRegularizerL1 > 0 {
		for _, bias := range layer.Biases {
			regularizationLoss += layer.BiasRegularizerL1 * math.Abs(bias)
		}
	}

	// L2 regularization - biases
	if layer.BiasRegularizerL2 > 0 {
		for _, bias := range layer.Biases {
			regularizationLoss += layer.BiasRegularizerL2 * bias * bias
		}
	}

	return regularizationLoss
}

func NewActivationSoftmaxLoss() *ActivationSoftmaxLoss {

	var loss NormalLoss
	return &ActivationSoftmaxLoss{ActivationSoftMax: NewActivationSoftMax(), Loss: loss}
}

func (ASL *ActivationSoftmaxLoss) Forward(input tensor.Tensor, yTrue tensor.Tensor) float64 {
	ASL.ActivationSoftMax.Forward(input)
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
	ASL.DInputs = dinputs.Clone().(tensor.Tensor)
}

func NewBinaryCrossentropy() *BinaryCrossentropyLoss {

	var loss NormalLoss
	return &BinaryCrossentropyLoss{Loss: loss}
}

func (BCL *BinaryCrossentropyLoss) Forward(yPred tensor.Tensor, yTrue tensor.Tensor) float64 {

	YPredClipped := clip(yPred, 1e-7, 1-1e-7)

	predictedValsBacking := YPredClipped.Data().([]float64)

	yTrueBk := yTrue.Data().([]float64)
	sampleLossesBk := make([]float64, yPred.Shape()[0])
	for i := range predictedValsBacking {
		sampleLossesBk[i] = -(yTrueBk[i] * math.Log(predictedValsBacking[i])) +
			((1 - yTrueBk[i]) * math.Log(1-predictedValsBacking[i]))
	}
	samplpeLoss := tensor.New(tensor.WithShape(yTrue.Shape()...), tensor.WithBacking(sampleLossesBk))
	sampleLossesBk = meanAlongLastAxis(TensorToFloat64Slice(samplpeLoss))

	BCL.DataLoss = mean(sampleLossesBk)

	return BCL.DataLoss

}

func (BCL *BinaryCrossentropyLoss) Backward(dvalues tensor.Tensor, yTrue tensor.Tensor) {

	var err error

	sample := dvalues.Shape()[0]
	outputs := dvalues.Shape()[1]
	clippedDvalues := clip(dvalues, 1e-7, 1-1e-7)
	clippedDvaluesBk := clippedDvalues.Data().([]float64)

	dinputs := clippedDvalues.Clone().(tensor.Tensor)
	yTrueBk := yTrue.Data().([]float64)
	dInputsBk := make([]float64, len(clippedDvaluesBk))

	for i := range clippedDvaluesBk {
		dInputsBk[i] = -((yTrueBk[i] / clippedDvaluesBk[i]) -
			((1 - yTrueBk[i]) / (1 - clippedDvaluesBk[i]))) / float64(outputs)
		dInputsBk[i] = dInputsBk[i] / float64(sample)
	}

	BCL.DInputs = tensor.New(tensor.WithShape(dinputs.Shape()...), tensor.WithBacking(dInputsBk))
	//fmt.Println(dInputsBk[120])
	handleErr(err)
}

func NewMeanSquaredErrorLoss() *MeanSquaredErrorLoss {

	var loss NormalLoss
	return &MeanSquaredErrorLoss{Loss: loss}
}

func (MSE *MeanSquaredErrorLoss) Forward(yPred tensor.Tensor, yTrue tensor.Tensor) float64 {

	predictedValsBacking := yPred.Data().([]float64)

	yTrueBk := yTrue.Data().([]float64)
	sampleLossesBk := make([]float64, yPred.Shape()[0])
	for i := range predictedValsBacking {
		sampleLossesBk[i] = math.Pow(yTrueBk[i]-predictedValsBacking[i], 2)
	}
	samplpeLoss := tensor.New(tensor.WithShape(yTrue.Shape()...), tensor.WithBacking(sampleLossesBk))
	sampleLossesBk = meanAlongLastAxis(TensorToFloat64Slice(samplpeLoss))

	MSE.DataLoss = mean(sampleLossesBk)

	return MSE.DataLoss

}

func (MSE *MeanSquaredErrorLoss) Backward(dvalues tensor.Tensor, yTrue tensor.Tensor) {

	var err error

	sample := dvalues.Shape()[0]
	outputs := dvalues.Shape()[1]
	//clippedDvalues := clip(dvalues, 1e-7, 1-1e-7)
	//clippedDvaluesBk := dvalues.Data().([]float64)

	dinputs := dvalues.Clone().(tensor.Tensor)
	dvaluesBk := dvalues.Data().([]float64)
	yTrueBk := yTrue.Data().([]float64)
	dInputsBk := make([]float64, len(dvaluesBk))

	for i := range dvaluesBk {
		dInputsBk[i] = (-2 * (yTrueBk[i] - dvaluesBk[i])) / float64(outputs)
		dInputsBk[i] = dInputsBk[i] / float64(sample)
	}

	MSE.DInputs = tensor.New(tensor.WithShape(dinputs.Shape()...), tensor.WithBacking(dInputsBk))
	handleErr(err)
}
func NewMeanAbsoluteErrorLoss() *MeanAbsoluteErrorLoss {

	var loss NormalLoss
	return &MeanAbsoluteErrorLoss{Loss: loss}
}

func (MAE *MeanAbsoluteErrorLoss) Forward(yPred tensor.Tensor, yTrue tensor.Tensor) float64 {

	predictedValsBacking := yPred.Data().([]float64)

	yTrueBk := yTrue.Data().([]float64)
	sampleLossesBk := make([]float64, yPred.Shape()[0])
	for i := range predictedValsBacking {
		sampleLossesBk[i] = math.Abs(yTrueBk[i] - predictedValsBacking[i])
	}
	samplpeLoss := tensor.New(tensor.WithShape(yTrue.Shape()...), tensor.WithBacking(sampleLossesBk))
	sampleLossesBk = meanAlongLastAxis(TensorToFloat64Slice(samplpeLoss))

	MAE.DataLoss = mean(sampleLossesBk)

	return MAE.DataLoss

}

func (MAE *MeanAbsoluteErrorLoss) Backward(dvalues tensor.Tensor, yTrue tensor.Tensor) {

	var err error

	sample := dvalues.Shape()[0]
	outputs := dvalues.Shape()[1]
	//clippedDvalues := clip(dvalues, 1e-7, 1-1e-7)
	//clippedDvaluesBk := dvalues.Data().([]float64)

	dinputs := dvalues.Clone().(tensor.Tensor)
	dvaluesBk := dvalues.Data().([]float64)
	yTrueBk := yTrue.Data().([]float64)
	dInputsBk := make([]float64, len(dvaluesBk))
	dInputsBk = signSubtract(yTrueBk, dvaluesBk)
	for i := range dvaluesBk {
		dInputsBk[i] = dInputsBk[i] / float64(outputs)
		dInputsBk[i] = dInputsBk[i] / float64(sample)
	}

	MAE.DInputs = tensor.New(tensor.WithShape(dinputs.Shape()...), tensor.WithBacking(dInputsBk))
	handleErr(err)
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

func Accuracy(predictions []int, y []float64) float64 {
	var correct int
	for i, pred := range predictions {
		if float64(pred) == y[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(y))
}

func meanAlongLastAxis(outputs [][]float64) []float64 {
	means := make([]float64, len(outputs))
	for i, row := range outputs {
		sum := 0.0
		for _, value := range row {
			sum += value
		}
		means[i] = sum / float64(len(row))
	}
	return means
}

func clip(yPred tensor.Tensor, lower, upper float64) tensor.Tensor {
	yPredBk := yPred.Data().([]float64)
	clipped := make([]float64, len(yPredBk))
	for i, val := range yPredBk {
		if val < lower {
			clipped[i] = lower
		} else if val > upper {
			clipped[i] = upper
		} else {
			clipped[i] = val
		}
	}
	return tensor.New(tensor.WithShape(yPred.Shape()...), tensor.WithBacking(clipped))
}

func Threshold(t tensor.Tensor, threshold float64) *tensor.Dense {

	preds := t.Clone().(*tensor.Dense)
	it := t.Iterator()
	//fmt.Println(preds.Data().([]float64))
	for _, errr := it.Start(); errr == nil; _, errr = it.Next() {
		val, _ := t.At(it.Coord()...)
		if val == nil {
			continue
		}
		if val.(float64) > threshold {
			preds.SetAt(1.0, it.Coord()...)
		} else {
			preds.SetAt(0.0, it.Coord()...)
		}
	}
	return preds

}

func AccuracyByTensor(predictions, y *tensor.Dense) float64 {

	// Initialize a variable to count correct predictions
	correct := 0.0
	predictionBk := TensorToFloat64Slice(predictions)
	yBk := TensorToFloat64Slice(y)

	// Iterate over predictions and true labels

	if y.Shape()[1] == 1 {
		for i := range predictionBk {
			//for j := range predictionBk[i] {
			// If prediction equals true label, increment correct count
			if predictionBk[i][0] == yBk[i][0] {
				correct++
			}
			//}
		}
	}

	// Calculate accuracy
	total := float64(len(predictionBk) * len(yBk[0]))
	accuracy := correct / total

	return accuracy
}

func AccuracyByRegression(preds, yTrue tensor.Tensor, accuracyPrecision float64) float64 {
	count := 0
	predictions := preds.Data().([]float64)
	y := yTrue.Data().([]float64)
	if yTrue.Shape()[1] == 1 {
		for i := range predictions {
			if math.Abs(predictions[i]-y[i]) < accuracyPrecision {
				count++
			}
		}
	}

	return float64(count) / float64(len(predictions))
}

func signSubtract(yTrue, dValues []float64) []float64 {
	result := make([]float64, len(yTrue))
	for i := 0; i < len(yTrue); i++ {
		if math.Signbit(yTrue[i] - dValues[i]) {
			result[i] = -1
		} else if yTrue[i] == dValues[i] {
			result[i] = 0
		} else {
			result[i] = 1
		}
	}
	return result
}

func StdDev(y []float64) float64 {
	sum := 0.0
	for _, v := range y {
		sum += v
	}
	mean := sum / float64(len(y))

	variance := 0.0
	for _, v := range y {
		variance += math.Pow(v-mean, 2)
	}
	variance /= float64(len(y))

	return math.Sqrt(variance)
}
