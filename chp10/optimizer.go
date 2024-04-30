package chp10

import (
	"aiPr/chp3"
	"fmt"
	"gorgonia.org/tensor"
	"math"
)

type OptimizerSGD struct {
	LearningRate        float64
	CurrentLearningRate float64
	Decay               float64
	Iteration           float64
	Momentum            float64
}

func NewOptimizerSGD(learningRate, decay, momentum float64) OptimizerSGD {
	return OptimizerSGD{
		LearningRate:        learningRate,
		CurrentLearningRate: learningRate,
		Decay:               decay,
		Iteration:           0,
		Momentum:            momentum,
	}
}

func (o *OptimizerSGD) UpdateParams(layer chp3.LayerDense) chp3.LayerDense {

	var err error
	var weightUpdates *tensor.Dense
	var biasUpdates []float64
	if o.Momentum > 0 {
		if layer.WeightMomentums == nil {
			weightMomentums := tensor.New(
				tensor.WithShape(layer.Weights.Shape()...),
				tensor.WithBacking(tensor.Random(tensor.Float64, layer.Weights.Shape()[0]*layer.Weights.Shape()[1])),
			)
			layer.WeightMomentums, err = weightMomentums.MulScalar(0.01, false)
			handleErr(err)
			layer.BiasMomentums = make([]float64, layer.Weights.Shape()[1])
			for i := range layer.BiasMomentums {
				layer.BiasMomentums[i] = 0
			}
		}

		//
		weightUpdatesMo := tensor.New(
			tensor.WithShape(layer.WeightMomentums.Shape()...),
			tensor.WithBacking(layer.WeightMomentums.Data()),
		)
		weightUpdatesDw := tensor.New(
			tensor.WithShape(layer.DWeights.Shape()...),
			tensor.WithBacking(layer.DWeights.Data()),
		)
		weightUpdatesMo, err = weightUpdatesMo.MulScalar(o.Momentum, false)
		handleErr(err)
		weightUpdatesDw, err = weightUpdatesDw.MulScalar(o.CurrentLearningRate, false)
		handleErr(err)
		weightUpdates, err = weightUpdatesMo.Sub(weightUpdatesDw)
		handleErr(err)
		layer.WeightMomentums = weightUpdates

		biasUpdates = make([]float64, len(layer.BiasMomentums))
		for i, _ := range biasUpdates {
			biasUpdates[i] =
				(o.Momentum * layer.BiasMomentums[i]) -
					(o.CurrentLearningRate * layer.DBiases[i])
			layer.BiasMomentums = biasUpdates
		}

	} else {
		weightUpdates = tensor.New(tensor.WithShape(layer.DWeights.Shape()...), tensor.WithBacking(layer.DWeights.Data()))
		weightUpdates, _ = weightUpdates.MulScalar(-o.CurrentLearningRate, false)
		biasUpdates = make([]float64, len(layer.DBiases))
		for i := range layer.Biases {
			biasUpdates[i] = -o.CurrentLearningRate * layer.DBiases[i]
		}
	}

	layer.Weights, _ = tensor.Add(layer.Weights, weightUpdates)
	for i := range layer.Biases {
		layer.Biases[i] += biasUpdates[i]
	}

	return layer

}

func (o *OptimizerSGD) PreUpdateParams() {
	o.CurrentLearningRate = o.LearningRate * (1.0 / (1.0 + o.Decay*o.Iteration))
}

func (o *OptimizerSGD) PostUpdateParams() {
	o.Iteration++
}

type OptimizerAdagrad struct {
	LearningRate        float64
	CurrentLearningRate float64
	Decay               float64
	Iteration           float64
	Epsilon             float64
}

func NewOptimizerAdagrad(learningRate, decay, epsilon float64) OptimizerAdagrad {
	return OptimizerAdagrad{
		LearningRate:        learningRate,
		CurrentLearningRate: learningRate,
		Decay:               decay,
		Iteration:           0,
		Epsilon:             epsilon,
	}
}

func (oa *OptimizerAdagrad) UpdateParams(layer chp3.LayerDense) chp3.LayerDense {

	var err error
	if oa.Epsilon == 0 {
		oa.Epsilon = 1e-7
	}
	// If layer does not contain cache arrays,
	// create them filled with zeros
	if layer.WeightCache == nil {
		shape := layer.Weights.Shape()
		weightCache := tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(shape...))

		layer.WeightCache = weightCache

		biasCache := make([]float64, len(layer.Biases))
		layer.BiasCache = biasCache
	}

	// Update cache with squared current gradients
	squaredDWeights, err := tensor.Square(layer.DWeights)

	squaredDBiases, err := tensor.Square(tensor.New(tensor.WithShape(len(layer.DBiases)), tensor.WithBacking(layer.DBiases)))

	layer.WeightCache, err = tensor.Add(layer.WeightCache, squaredDWeights)

	for i := range layer.BiasCache {
		layer.BiasCache[i] += squaredDBiases.Data().([]float64)[i]
	}

	// Vanilla SGD parameter update + normalization
	// with square rooted cache
	cacheWeightSqT, _ := tensor.Sqrt(layer.WeightCache)
	cacheWeightSq := tensor.New(tensor.WithShape(cacheWeightSqT.Shape()...), tensor.WithBacking(cacheWeightSqT.Data()))
	cacheWeightByEp, _ := cacheWeightSq.AddScalar(oa.Epsilon, false)

	dweightByRate := tensor.New(tensor.WithShape(layer.DWeights.Shape()...), tensor.WithBacking(layer.DWeights.Data()))
	dweightByRate, err = dweightByRate.MulScalar(-oa.CurrentLearningRate, false)

	weightUpdates, err := tensor.Div(
		dweightByRate,
		cacheWeightByEp,
	)

	layer.Weights, err = tensor.Add(layer.Weights, weightUpdates)

	for i := range layer.Biases {
		layer.Biases[i] += (-oa.CurrentLearningRate * layer.DBiases[i]) /
			(math.Sqrt(layer.BiasCache[i]) + oa.Epsilon)
	}

	handleErr(err)
	return layer
}

func (oa *OptimizerAdagrad) PreUpdateParams() {
	oa.CurrentLearningRate = oa.LearningRate * (1.0 / (1.0 + oa.Decay*oa.Iteration))
}

func (oa *OptimizerAdagrad) PostUpdateParams() {
	oa.Iteration++
}

type OptimizerRMSprop struct {
	LearningRate        float64
	CurrentLearningRate float64
	Decay               float64
	Iteration           float64
	Epsilon             float64
	Rho                 float64
}

func NewOptimizerRMSprop(learningRate, decay, epsilon, rho float64) OptimizerRMSprop {
	return OptimizerRMSprop{
		LearningRate:        learningRate,
		CurrentLearningRate: learningRate,
		Decay:               decay,
		Iteration:           0,
		Epsilon:             epsilon,
		Rho:                 rho,
	}
}

func (or *OptimizerRMSprop) UpdateParams(layer chp3.LayerDense) chp3.LayerDense {

	var err error

	if or.Epsilon == 0 {
		or.Epsilon = 1e-7
	}

	// If layer does not contain cache arrays,
	// create them filled with zeros
	if layer.WeightCache == nil {
		shape := layer.Weights.Shape()
		weightCache := tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(shape...))
		layer.WeightCache = weightCache

		biasCache := make([]float64, len(layer.Biases))
		layer.BiasCache = biasCache
	}

	// Update cache with squared current gradients
	squaredDWeights, err := tensor.Square(layer.DWeights)
	squaredDWeightsDense := tensor.New(tensor.WithShape(squaredDWeights.Shape()...), tensor.WithBacking(squaredDWeights.Data()))
	handleErr(err)
	squaredDWeightsDenseBtRho, err := squaredDWeightsDense.MulScalar(1-or.Rho, false)
	handleErr(err)

	WeightCacheDens := tensor.New(tensor.WithShape(layer.WeightCache.Shape()...), tensor.WithBacking(layer.WeightCache.Data()))
	handleErr(err)
	WeightCacheDensByRho, err := WeightCacheDens.MulScalar(or.Rho, false)
	handleErr(err)

	layer.WeightCache, err = tensor.Add(
		WeightCacheDensByRho,
		squaredDWeightsDenseBtRho,
	)
	handleErr(err)

	for i := range layer.BiasCache {
		layer.BiasCache[i] = or.Rho*layer.BiasCache[i] + (1-or.Rho)*math.Pow(layer.DBiases[i], 2)
	}

	// Vanilla SGD parameter update + normalization
	// with square rooted cache
	cacheWeightSqT, err := tensor.Sqrt(layer.WeightCache)
	handleErr(err)
	cacheWeightSq := tensor.New(tensor.WithShape(cacheWeightSqT.Shape()...), tensor.WithBacking(cacheWeightSqT.Data()))
	cacheWeightByEp, err := cacheWeightSq.AddScalar(or.Epsilon, false)
	handleErr(err)

	dweightByRate := tensor.New(tensor.WithShape(layer.DWeights.Shape()...), tensor.WithBacking(layer.DWeights.Data()))
	dweightByRate, err = dweightByRate.MulScalar(-or.CurrentLearningRate, false)
	handleErr(err)

	weightUpdates, err := tensor.Div(
		dweightByRate,
		cacheWeightByEp,
	)
	handleErr(err)

	layer.Weights, err = tensor.Add(layer.Weights, weightUpdates)
	handleErr(err)

	for i := range layer.Biases {
		layer.Biases[i] += (-or.CurrentLearningRate * layer.DBiases[i]) /
			(math.Sqrt(layer.BiasCache[i]) + or.Epsilon)
	}

	return layer
}

func (or *OptimizerRMSprop) PreUpdateParams() {
	or.CurrentLearningRate = or.LearningRate * (1.0 / (1.0 + or.Decay*or.Iteration))
}

func (or *OptimizerRMSprop) PostUpdateParams() {
	or.Iteration++
}

type OptimizerAdam struct {
	LearningRate        float64
	CurrentLearningRate float64
	Decay               float64
	Iteration           float64
	Epsilon             float64
	Beta1               float64
	Beta2               float64
}

func NewOptimizerAdam(learningRate, decay, epsilon, beta1, beta2 float64) OptimizerAdam {
	return OptimizerAdam{
		LearningRate:        learningRate,
		CurrentLearningRate: learningRate,
		Decay:               decay,
		Iteration:           0,
		Epsilon:             epsilon,
		Beta1:               beta1,
		Beta2:               beta2,
	}
}

func (oad *OptimizerAdam) UpdateParams(layer chp3.LayerDense) chp3.LayerDense {

	var weightMomentumCorrected, weightCacheCorrected tensor.Tensor
	var biasMomentumCorrected, biasCacheCorrected []float64
	var err error

	// If layer does not contain cache arrays, create them filled with zeros
	if layer.WeightMomentums == nil {
		weightMomentum := tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(layer.Weights.Shape()...))
		layer.WeightMomentums = weightMomentum
		layer.WeightCache = tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(layer.Weights.Shape()...))
		layer.BiasMomentums = make([]float64, len(layer.Biases))
		layer.BiasCache = make([]float64, len(layer.Biases))
	}

	// Update momentum with current gradients

	WeightMomentumsDens := tensor.New(tensor.WithShape(layer.WeightMomentums.Shape()...), tensor.WithBacking(layer.WeightMomentums.Data()))
	WeightMomentumsAdded, err := WeightMomentumsDens.AddScalar(oad.Beta1, false)
	handleErr(err)
	DWeightsDense := tensor.New(tensor.WithShape(layer.DWeights.Shape()...), tensor.WithBacking(layer.DWeights.Data()))
	DWeightsDenseMul, err := DWeightsDense.MulScalar(1-oad.Beta1, false)
	handleErr(err)

	layer.WeightMomentums, err = tensor.Add(
		WeightMomentumsAdded,
		DWeightsDenseMul,
	)
	handleErr(err)

	for i, _ := range layer.DBiases {
		layer.BiasMomentums[i] = (oad.Beta1 * layer.BiasMomentums[i]) +
			(1-oad.Beta1)*layer.DBiases[i]
	}

	// Get corrected momentum

	WeightMomentumsDens2 := tensor.New(tensor.WithShape(layer.WeightMomentums.Shape()...), tensor.WithBacking(layer.WeightMomentums.Data()))
	weightMomentumCorrected, err = WeightMomentumsDens2.DivScalar(math.Pow(1-oad.Beta1, oad.Iteration+1), false)
	handleErr(err)

	biasMomentumCorrected = make([]float64, len(layer.BiasMomentums))

	for i, _ := range layer.BiasMomentums {
		biasMomentumCorrected[i] = layer.BiasMomentums[i] / (math.Pow(1-oad.Beta1, oad.Iteration+1))
	}

	handleErr(err)

	// Update cache with squared current gradients

	dweightSquare, err := tensor.Square(layer.DWeights)
	dweightSquareDens := tensor.New(tensor.WithShape(dweightSquare.Shape()...), tensor.WithBacking(dweightSquare.Data()))
	dweightSquareMul, err := dweightSquareDens.MulScalar(1-oad.Beta2, false)
	weightCacheDens := tensor.New(tensor.WithShape(layer.WeightCache.Shape()...), tensor.WithBacking(layer.WeightCache.Data()))
	weightCacheMul, err := weightCacheDens.MulScalar(oad.Beta2, false)
	handleErr(err)
	layer.WeightCache, err = tensor.Add(
		dweightSquareMul,
		weightCacheMul,
	)

	handleErr(err)

	layer.BiasCache = make([]float64, len(layer.DBiases))

	for i, _ := range layer.DBiases {
		layer.BiasCache[i] = (layer.BiasCache[i] * oad.Beta2) + ((1 - oad.Beta2) * math.Pow(layer.DBiases[i], 2))
	}

	// Get corrected cache

	weightCacheCorrectedDens2 := tensor.New(tensor.WithShape(layer.WeightCache.Shape()...), tensor.WithBacking(layer.WeightCache.Data()))
	weightCacheCorrected, err = weightCacheCorrectedDens2.DivScalar(math.Pow(1-oad.Beta2, oad.Iteration+1), false)
	handleErr(err)

	biasCacheCorrected = make([]float64, len(layer.BiasCache))

	for i, _ := range layer.BiasCache {
		biasCacheCorrected[i] = layer.BiasCache[i] / (math.Pow(1-oad.Beta2, oad.Iteration+1))
	}

	handleErr(err)

	// Vanilla SGD parameter update + normalization with square rooted cache

	WeightMomentumsDens3 := tensor.New(tensor.WithShape(weightMomentumCorrected.Shape()...), tensor.WithBacking(weightMomentumCorrected.Data()))
	weightMomentumCorrectedMul, err := WeightMomentumsDens3.MulScalar(-oad.CurrentLearningRate, false)

	weightCacheCorrectedSqt, err := tensor.Sqrt(weightCacheCorrected)
	handleErr(err)
	weightCacheCorrectedSqtDens := tensor.New(tensor.WithShape(weightCacheCorrectedSqt.Shape()...), tensor.WithBacking(weightCacheCorrectedSqt.Data()))
	weightCacheCorrectedAdded, err := weightCacheCorrectedSqtDens.AddScalar(oad.Epsilon, false)
	handleErr(err)

	weightUpdates, err := tensor.Div(
		weightMomentumCorrectedMul,
		weightCacheCorrectedAdded,
	)

	handleErr(err)

	layer.Weights, err = tensor.Add(layer.Weights, weightUpdates)
	//fmt.Println(layer.Weights.Data())
	handleErr(err)

	for i := range layer.Biases {
		layer.Biases[i] += (-oad.CurrentLearningRate * biasMomentumCorrected[i]) /
			(math.Sqrt(biasCacheCorrected[i]) + oad.Epsilon)
	}
	handleErr(err)
	//fmt.Println(layer.Biases)
	return layer

}
func (oad *OptimizerAdam) UpdateParams2(layer chp3.LayerDense) chp3.LayerDense {

	var weightMomentumCorrected, weightCacheCorrected tensor.Tensor
	var biasMomentumCorrected, biasCacheCorrected []float64
	var err error

	// If layer does not contain cache arrays, create them filled with zeros
	if layer.WeightMomentums == nil {
		weightMomentum := tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(layer.Weights.Shape()...))
		layer.WeightMomentums = weightMomentum
		layer.WeightCache = tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(layer.Weights.Shape()...))
		layer.BiasMomentums = make([]float64, len(layer.Biases))
		layer.BiasCache = make([]float64, len(layer.Biases))
	}

	// Update momentum with current gradients

	WeightMomentumsDens := tensor.New(tensor.WithShape(layer.WeightMomentums.Shape()...), tensor.WithBacking(layer.WeightMomentums.Data()))
	WeightMomentumsAdded, err := WeightMomentumsDens.AddScalar(oad.Beta1, false)
	handleErr(err)
	DWeightsDense := tensor.New(tensor.WithShape(layer.DWeights.Shape()...), tensor.WithBacking(layer.DWeights.Data()))
	DWeightsDenseMul, err := DWeightsDense.MulScalar(1-oad.Beta1, false)
	handleErr(err)

	layer.WeightMomentums, err = tensor.Add(
		WeightMomentumsAdded,
		DWeightsDenseMul,
	)
	handleErr(err)

	for i, _ := range layer.DBiases {
		layer.BiasMomentums[i] = (oad.Beta1 * layer.BiasMomentums[i]) +
			(1-oad.Beta1)*layer.DBiases[i]
	}

	// Get corrected momentum

	WeightMomentumsDens2 := tensor.New(tensor.WithShape(layer.WeightMomentums.Shape()...), tensor.WithBacking(layer.WeightMomentums.Data()))
	x := 1 - oad.Beta1
	y := oad.Iteration + 1
	weightMomentumCorrected, err = WeightMomentumsDens2.DivScalar(math.Pow(x, y), false)
	fmt.Println(layer.WeightMomentums)
	//weightMomentumCorrected = layer.WeightMomentums
	handleErr(err)

	biasMomentumCorrected = make([]float64, len(layer.BiasMomentums))

	for i, _ := range layer.BiasMomentums {
		biasMomentumCorrected[i] = layer.BiasMomentums[i] / (math.Pow(x, y))
	}

	handleErr(err)
	// Update cache with squared current gradients

	dweightSquare, err := tensor.Square(layer.DWeights)
	dweightSquareDens := tensor.New(tensor.WithShape(dweightSquare.Shape()...), tensor.WithBacking(dweightSquare.Data()))
	dweightSquareMul, err := dweightSquareDens.MulScalar(1-oad.Beta2, false)
	weightCacheDens := tensor.New(tensor.WithShape(layer.WeightCache.Shape()...), tensor.WithBacking(layer.WeightCache.Data()))
	weightCacheMul, err := weightCacheDens.MulScalar(oad.Beta2, false)
	handleErr(err)
	layer.WeightCache, err = tensor.Add(
		dweightSquareMul,
		weightCacheMul,
	)

	handleErr(err)

	layer.BiasCache = make([]float64, len(layer.DBiases))

	for i, _ := range layer.DBiases {
		layer.BiasCache[i] = (layer.BiasCache[i] * oad.Beta2) + ((1 - oad.Beta2) * math.Pow(layer.DBiases[i], 2))
	}

	// Get corrected cache

	weightCacheCorrectedDens2 := tensor.New(tensor.WithShape(layer.WeightCache.Shape()...), tensor.WithBacking(layer.WeightCache.Data()))
	weightCacheCorrected, err = weightCacheCorrectedDens2.DivScalar(math.Pow(1-oad.Beta2, oad.Iteration+1), false)
	handleErr(err)
	biasCacheCorrected = make([]float64, len(layer.BiasCache))
	for i, _ := range layer.BiasCache {
		biasCacheCorrected[i] = layer.BiasCache[i] / (math.Pow(1-oad.Beta2, oad.Iteration+1))
	}

	handleErr(err)
	// Vanilla SGD parameter update + normalization with square rooted cache
	weightMomentumCorrectedDens := tensor.New(tensor.WithShape(weightMomentumCorrected.Shape()...), tensor.WithBacking(weightMomentumCorrected.Data()))
	weightMomentumCorrectedMul, err := weightMomentumCorrectedDens.MulScalar(-oad.CurrentLearningRate, false)
	handleErr(err)
	weightCacheCorrectedSqt, err := tensor.Sqrt(weightCacheCorrected)
	handleErr(err)
	weightCacheCorrectedSqtDens := tensor.New(tensor.WithShape(weightCacheCorrectedSqt.Shape()...), tensor.WithBacking(weightCacheCorrectedSqt.Data()))
	weightCacheCorrectedSqtAdd, err := weightCacheCorrectedSqtDens.AddScalar(oad.Epsilon, false)
	handleErr(err)
	weightUpdates, err := tensor.Div(weightMomentumCorrectedMul, weightCacheCorrectedSqtAdd)
	handleErr(err)
	layer.Weights, err = tensor.Add(layer.Weights, weightUpdates)
	handleErr(err)

	for i := range layer.Biases {
		layer.Biases[i] += (-oad.CurrentLearningRate * biasMomentumCorrected[i]) /
			(math.Sqrt(biasCacheCorrected[i]) + oad.Epsilon)
	}
	handleErr(err)
	//fmt.Println(layer.Biases)
	return layer

}

func (oad *OptimizerAdam) PreUpdateParams() {
	oad.CurrentLearningRate = oad.LearningRate * (1.0 / (1.0 + oad.Decay*oad.Iteration))
}

func (oad *OptimizerAdam) PostUpdateParams() {
	oad.Iteration++
}

func handleErr(er error) {
	if er != nil {
		fmt.Println("error", er)
	}
}
