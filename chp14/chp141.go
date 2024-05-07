package chp14

import (
	"fmt"
	"gorgonia.org/tensor"
)

func RunRegulizerFunc1() {

	///////forwarding
	//layer 1
	dense1 := ml.NewLayerDense(2, 64, 0, 5e-4, 0, 5e-4)
	activation1 := ml.NewActivationReLU()
	//layer 2
	dense2 := ml.NewLayerDense(64, 3)
	lossActiovatoin := ml.NewActivationSoftmaxLoss()
	optimizer := ml.NewOptimizerAdam(0.05, 5e-7, 1e-7, 0.9, 0.999)
	//optimizer2 := NewOptimizerAdam(0.02, 5e-7, 1e-7, 0.009, 0.000999)
	var accuracy = float64(0)
	for i := 1; i <= 5000; i++ {

		dense1.Forward(ml.X3)
		activation1.Forward(dense1.Output)
		dense2.Forward(activation1.Output)
		lossActiovatoin.Forward(dense2.Output, ml.Yval3)

		/////// calculate accuracy
		predictions, _ := tensor.Argmax(lossActiovatoin.ActivationSoftMax.Output, 1)
		accuracy = ml.Accuracy(predictions.Data().([]int), ml.Yval3.Data().([]float64))

		//calc regularize loss
		regularizationLoss := lossActiovatoin.Loss.RegularizationLoss(dense1) + lossActiovatoin.Loss.RegularizationLoss(dense2)
		loss := lossActiovatoin.Loss.DataLoss + regularizationLoss

		////// backward
		//fmt.Println(lossActiovatoin.ActivationSoftMax.Output, lossActiovatoin.Loss.DInputs)

		lossActiovatoin.Backward(lossActiovatoin.ActivationSoftMax.Output, ml.Yval3)

		dense2.Backward(lossActiovatoin.DInputs)
		activation1.Backward(dense2.DInput)
		dense1.Backward(activation1.DInput)

		optimizer.PreUpdateParams()
		dense1 = optimizer.UpdateParams(dense1)
		dense2 = optimizer.UpdateParams(dense2)
		optimizer.PostUpdateParams()

		if i%100 == 0 {
			fmt.Println("step", i)
			fmt.Println("loss", lossActiovatoin.Loss.DataLoss)
			fmt.Println("loss with penalty (regulazrization)", loss)
			fmt.Println("acc", accuracy)
			fmt.Println("CurrentLearningRate", optimizer.CurrentLearningRate)
		}

	}

	//fmt.Println("d1 w", dense1.Weights.Data().([]float64))
	//fmt.Println("d2 w", dense2.Weights.Data().([]float64))
	//fmt.Println("d1 b", dense1.Biases)
	//fmt.Println("d2 b", dense2.Biases)

	dense1.Forward(ml.X)
	activation1.Forward(dense1.Output)
	dense2.Forward(activation1.Output)
	lossActiovatoin.Forward(dense2.Output, ml.Yval)

	predictions, _ := tensor.Argmax(lossActiovatoin.ActivationSoftMax.Output, 1)
	accuracy = ml.Accuracy(predictions.Data().([]int), ml.Yval.Data().([]float64))

	fmt.Println("loss for new data", lossActiovatoin.Loss.DataLoss)
	fmt.Println("acc for new data", accuracy)

	return
}

func RunRegulizerIncreaseNodesFunc1() {

	///////forwarding
	//layer 1
	dense1 := ml.NewLayerDense(2, 512, 0, 5e-4, 0, 5e-4)
	activation1 := ml.NewActivationReLU()
	//layer 2
	dense2 := ml.NewLayerDense(512, 3)
	lossActiovatoin := ml.NewActivationSoftmaxLoss()
	optimizer := ml.NewOptimizerAdam(0.05, 5e-7, 1e-7, 0.9, 0.999)
	//optimizer2 := NewOptimizerAdam(0.02, 5e-7, 1e-7, 0.009, 0.000999)
	var accuracy = float64(0)
	for i := 1; i <= 10000; i++ {

		dense1.Forward(ml.X)
		activation1.Forward(dense1.Output)
		dense2.Forward(activation1.Output)
		lossActiovatoin.Forward(dense2.Output, ml.Yval)

		/////// calculate accuracy
		predictions, _ := tensor.Argmax(lossActiovatoin.ActivationSoftMax.Output, 1)
		accuracy = ml.Accuracy(predictions.Data().([]int), ml.Yval.Data().([]float64))

		//calc regularize loss
		regularizationLoss := lossActiovatoin.Loss.RegularizationLoss(dense1) + lossActiovatoin.Loss.RegularizationLoss(dense2)
		loss := lossActiovatoin.Loss.DataLoss + regularizationLoss

		////// backward
		//fmt.Println(lossActiovatoin.ActivationSoftMax.Output, lossActiovatoin.Loss.DInputs)

		lossActiovatoin.Backward(lossActiovatoin.ActivationSoftMax.Output, ml.Yval)

		dense2.Backward(lossActiovatoin.DInputs)
		activation1.Backward(dense2.DInput)
		dense1.Backward(activation1.DInput)

		optimizer.PreUpdateParams()
		dense1 = optimizer.UpdateParams(dense1)
		dense2 = optimizer.UpdateParams(dense2)
		optimizer.PostUpdateParams()

		if i%100 == 0 {
			fmt.Println("step", i)
			fmt.Println("loss", lossActiovatoin.Loss.DataLoss)
			fmt.Println("loss with penalty (regulazrization)", loss)
			fmt.Println("acc", accuracy)
			fmt.Println("CurrentLearningRate", optimizer.CurrentLearningRate)
		}

	}

	//fmt.Println("d1 w", dense1.Weights.Data().([]float64))
	//fmt.Println("d2 w", dense2.Weights.Data().([]float64))
	//fmt.Println("d1 b", dense1.Biases)
	//fmt.Println("d2 b", dense2.Biases)

	dense1.Forward(ml.X3)
	activation1.Forward(dense1.Output)
	dense2.Forward(activation1.Output)
	lossActiovatoin.Forward(dense2.Output, ml.Yval3)

	predictions, _ := tensor.Argmax(lossActiovatoin.ActivationSoftMax.Output, 1)
	accuracy = ml.Accuracy(predictions.Data().([]int), ml.Yval3.Data().([]float64))

	fmt.Println("loss for new data", lossActiovatoin.Loss.DataLoss)
	fmt.Println("acc for new data", accuracy)

	return
}
