package chp10

import (
	"aiPr/chp3"
)

type OptimizerSGD struct {
	LearningRate float64
}

func NewOptimizerSGD(learningRate float64) OptimizerSGD {
	return OptimizerSGD{LearningRate: learningRate}
}

func (l *OptimizerSGD) UpdateParams(layer chp3.LayerDense) {

	return

}
