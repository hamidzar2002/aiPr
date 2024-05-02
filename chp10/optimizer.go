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
	Iterations          int
	Epsilon             float64
	Beta1               float64
	Beta2               float64
}

func NewOptimizerAdam(learningRate, decay, epsilon, beta1, beta2 float64) *OptimizerAdam {
	return &OptimizerAdam{
		LearningRate:        learningRate,
		CurrentLearningRate: learningRate,
		Decay:               decay,
		Iterations:          0,
		Epsilon:             epsilon,
		Beta1:               beta1,
		Beta2:               beta2,
	}
}

func (optimizer *OptimizerAdam) PreUpdateParams() {
	if optimizer.Decay != 0 {
		optimizer.CurrentLearningRate = optimizer.LearningRate / (1.0 + optimizer.Decay*float64(optimizer.Iterations))
	}
}

func (optimizer *OptimizerAdam) UpdateParams(layer chp3.LayerDense) chp3.LayerDense {
	var WeightMomentumsBk, WeightCacheBk []float64
	if layer.WeightMomentums == nil {
		WeightMomentumsBk = make([]float64, len(layer.Weights.Data().([]float64)))
		WeightCacheBk = make([]float64, len(layer.Weights.Data().([]float64)))
		layer.BiasMomentums = make([]float64, len(layer.Biases))
		layer.BiasCache = make([]float64, len(layer.Biases))
	} else {
		WeightMomentumsBk = layer.WeightMomentums.Data().([]float64)
		WeightCacheBk = layer.WeightCache.Data().([]float64)

	}

	for i, _ := range layer.Weights.Data().([]float64) {
		// Update momentum with current gradients
		WeightMomentumsBk[i] = optimizer.Beta1*WeightMomentumsBk[i] + (1-optimizer.Beta1)*layer.DWeights.Data().([]float64)[i]
		// Get corrected momentum
		// optimizer.Iteration is 0 at first pass
		// and we need to start with 1 here
		weightMomentumCorrected := WeightMomentumsBk[i] / (1 - math.Pow(optimizer.Beta1, float64(optimizer.Iterations+1)))
		// Update cache with squared current gradients
		WeightCacheBk[i] = optimizer.Beta2*WeightCacheBk[i] + (1-optimizer.Beta2)*math.Pow(layer.DWeights.Data().([]float64)[i], 2)
		// Get corrected cache
		weightCacheCorrected := WeightCacheBk[i] / (1 - math.Pow(optimizer.Beta2, float64(optimizer.Iterations+1)))
		// Vanilla SGD parameter update + normalization with square rooted cache
		//if weightCacheCorrected > 0 {
		layer.Weights.Data().([]float64)[i] += (-optimizer.CurrentLearningRate * weightMomentumCorrected) / (math.Sqrt(math.Abs(weightCacheCorrected)) + optimizer.Epsilon)
		//}
	}

	layer.WeightMomentums = tensor.New(tensor.WithShape(layer.Weights.Shape()...), tensor.WithBacking(WeightMomentumsBk))
	layer.WeightCache = tensor.New(tensor.WithShape(layer.Weights.Shape()...), tensor.WithBacking(WeightCacheBk))

	for i, _ := range layer.Biases {
		// Update momentum with current gradients
		layer.BiasMomentums[i] = optimizer.Beta1*layer.BiasMomentums[i] + (1-optimizer.Beta1)*layer.DBiases[i]
		// Get corrected momentum
		// optimizer.Iteration is 0 at first pass
		// and we need to start with 1 here
		biasMomentumCorrected := layer.BiasMomentums[i] / (1 - math.Pow(optimizer.Beta1, float64(optimizer.Iterations+1)))
		// Update cache with squared current gradients
		layer.BiasCache[i] = optimizer.Beta2*layer.BiasCache[i] + (1-optimizer.Beta2)*math.Pow(layer.DBiases[i], 2)
		// Get corrected cache
		biasCacheCorrected := layer.BiasCache[i] / (1 - math.Pow(optimizer.Beta2, float64(optimizer.Iterations+1)))
		// Vanilla SGD parameter update + normalization with square rooted cache
		if (biasCacheCorrected) > 0 {
			layer.Biases[i] += (-optimizer.CurrentLearningRate * biasMomentumCorrected) / (math.Sqrt(math.Abs(biasCacheCorrected)) + optimizer.Epsilon)

		}
	}
	//fmt.Println(layer.Weights.Data().([]float64)[0])
	return layer
}

func (optimizer *OptimizerAdam) PostUpdateParams() {
	optimizer.Iterations++
}
func handleErr(er error) {
	if er != nil {
		fmt.Println("error", er)
	}
}
