package chp10

import (
	"fmt"
	"gorgonia.org/tensor"
)

func RunUpstreamTestFunc10() {

	///////forwarding
	//layer 1
	dense1 := ml.NewLayerDense(2, 64)
	activation1 := ml.NewActivationReLU()
	//layer 2
	dense2 := ml.NewLayerDense(64, 3)
	lossActiovatoin := ml.NewActivationSoftmaxLoss()
	optimizer := NewOptimizerSGD(0.7, 0, 0)
	var accuracy = float64(0)
	for i := 1; i <= 10000; i++ {

		dense1.Forward(ml.X)
		activation1.Forward(dense1.Output)
		dense2.Forward(activation1.Output)
		lossActiovatoin.Forward(dense2.Output, ml.Yval)

		/////// calculate accuracy
		predictions, _ := tensor.Argmax(lossActiovatoin.ActivationSoftMax.Output, 1)
		accuracy = ml.Accuracy(predictions.Data().([]int), ml.Yval.Data().([]float64))

		////// backward
		//fmt.Println(lossActiovatoin.ActivationSoftMax.Output, lossActiovatoin.Loss.DInputs)
		lossActiovatoin.Backward(lossActiovatoin.ActivationSoftMax.Output, ml.Yval)
		dense2.Backward(lossActiovatoin.DInputs)
		activation1.Backward(dense2.DInput)
		dense1.Backward(activation1.DInput)

		dense1 = optimizer.UpdateParams(dense1)
		dense2 = optimizer.UpdateParams(dense2)

		//fmt.Println("dense1.DWeights2", dense1.Weights)
		//fmt.Println("dense1.DBiases2", dense1.Biases)
		//fmt.Println("dense2.DWeights2", dense2.Weights)
		//fmt.Println("dense1.DBiases2", dense1.Biases)
	}
	fmt.Println("first Layer", dense1.Weights, dense1.Biases)
	fmt.Println("second Layer", dense2.Weights, dense2.Biases)
	fmt.Println("loss", lossActiovatoin.Loss.DataLoss)
	fmt.Println("acc", accuracy)
	return
}
