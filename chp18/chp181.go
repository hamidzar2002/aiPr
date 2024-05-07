package chp18

import (
	"aiPr/ml"
)

func RunModelTestFunc1() {

	var model ml.Model
	model.Add(ml.NewLayerDense(1, 64))
	model.Add(ml.NewActivationReLU())
	model.Add(ml.NewLayerDense(1, 64))
	model.Add(ml.NewLayerDense(1, 64))
	model.Set(ml.NewMeanSquaredErrorLoss(), ml.NewOptimizerAdam(0.01, 1e-3, 1e-7, 0.9, 0.999))
	model.Finalize()
	model.Train(ml.XR, ml.YR, 10000, 100)
	return
}
