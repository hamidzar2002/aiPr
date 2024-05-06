package chp17

import (
	"aiPr/chp10"
	"aiPr/chp3"
	"aiPr/chp4"
	"aiPr/chp5"
	"fmt"
	"gorgonia.org/tensor"
)

func RunLinearActFunc2() {

	///////forwarding
	//layer 1
	//dense1 := chp3.NewLayerDense(1, 64, 0, 5e-4, 0, 5e-4)
	dense1 := chp3.NewLayerDense(1, 64, 0, 0, 0, 0)
	activation1 := chp4.NewActivationReLU()

	//layer 2
	dense2 := chp3.NewLayerDense(64, 1)
	activation2 := chp4.NewActivationLinear()
	lossFunction := chp5.NewMeanSquaredErrorLoss()
	optimizer := chp10.NewOptimizerAdam(0.02, 5e-7, 1e-7, 0.9, 0.999)
	accuracyPrecision := chp5.StdDev(chp3.YR.Data().([]float64)) / 250
	//optimizer2 := NewOptimizerAdam(0.02, 5e-7, 1e-7, 0.009, 0.000999)
	var accuracy = float64(0)
	for i := 1; i <= 10001; i++ {

		dense1.Forward(chp3.XR)
		activation1.Forward(dense1.Output)
		dense2.Forward(activation1.Output)
		activation2.Forward(dense2.Output)
		dataLoss := lossFunction.Forward(activation2.Outputs, chp3.YR)
		//calc regularize loss
		regularizationLoss := lossFunction.Loss.RegularizationLoss(dense1) + lossFunction.Loss.RegularizationLoss(dense2)
		loss := dataLoss + regularizationLoss

		/////// calculate accuracy
		predictions := activation2.Outputs.(tensor.Tensor)

		ytens := chp3.YR.Clone().(*tensor.Dense)
		accuracy = chp5.AccuracyByRegression(predictions, ytens, accuracyPrecision)

		if i%100 == 0 {
			fmt.Println("step", i)
			fmt.Println("loss", dataLoss)
			fmt.Println("regulazrization pen", regularizationLoss)
			fmt.Println("loss with penalty (regulazrization)", loss)
			fmt.Println("acc", accuracy)
			fmt.Println("CurrentLearningRate", optimizer.CurrentLearningRate)

		}

		////// backward
		lossFunction.Backward(activation2.Outputs, chp3.YR)
		activation2.Backward(lossFunction.DInputs)
		dense2.Backward(activation2.DInputs)
		activation1.Backward(dense2.DInput)
		dense1.Backward(activation1.DInput)

		//optimization
		optimizer.PreUpdateParams()
		dense1 = optimizer.UpdateParams(dense1)
		dense2 = optimizer.UpdateParams(dense2)
		optimizer.PostUpdateParams()

	}

	return
}
func RunLinearActFunc3() {

	///////forwarding
	//layer 1
	//dense1 := chp3.NewLayerDense(1, 64, 0, 5e-4, 0, 5e-4)
	dense1 := chp3.NewLayerDense(1, 64, 0, 0, 0, 0)
	activation1 := chp4.NewActivationReLU()

	//layer 2
	dense2 := chp3.NewLayerDense(64, 64)
	activation2 := chp4.NewActivationReLU()
	dense3 := chp3.NewLayerDense(64, 1)
	activation3 := chp4.NewActivationLinear()
	lossFunction := chp5.NewMeanSquaredErrorLoss()
	optimizer := chp10.NewOptimizerAdam(0.01, 1e-3, 1e-7, 0.9, 0.999)
	accuracyPrecision := chp5.StdDev(chp3.YR.Data().([]float64)) / 250
	//optimizer2 := NewOptimizerAdam(0.02, 5e-7, 1e-7, 0.009, 0.000999)
	var accuracy = float64(0)
	for i := 1; i <= 10001; i++ {

		dense1.Forward(chp3.XR)
		activation1.Forward(dense1.Output)
		dense2.Forward(activation1.Output)
		activation2.Forward(dense2.Output)
		dense3.Forward(activation2.Output)
		activation3.Forward(dense3.Output)
		dataLoss := lossFunction.Forward(activation3.Outputs, chp3.YR)
		//calc regularize loss
		regularizationLoss := lossFunction.Loss.RegularizationLoss(dense1) + lossFunction.Loss.RegularizationLoss(dense2) + lossFunction.Loss.RegularizationLoss(dense3)
		loss := dataLoss + regularizationLoss

		/////// calculate accuracy
		predictions := activation3.Outputs.(tensor.Tensor)

		ytens := chp3.YR.Clone().(*tensor.Dense)
		accuracy = chp5.AccuracyByRegression(predictions, ytens, accuracyPrecision)

		if i%100 == 0 {
			fmt.Println("step", i)
			fmt.Println("loss", dataLoss)
			fmt.Println("regulazrization pen", regularizationLoss)
			fmt.Println("loss with penalty (regulazrization)", loss)
			fmt.Println("acc", accuracy)
			fmt.Println("CurrentLearningRate", optimizer.CurrentLearningRate)

		}

		////// backward
		lossFunction.Backward(activation3.Outputs, chp3.YR)
		activation3.Backward(lossFunction.DInputs)
		dense3.Backward(activation3.DInputs)
		activation2.Backward(dense3.DInput)
		dense2.Backward(activation2.DInput)
		activation1.Backward(dense2.DInput)
		dense1.Backward(activation1.DInput)

		//optimization
		optimizer.PreUpdateParams()
		dense1 = optimizer.UpdateParams(dense1)
		dense2 = optimizer.UpdateParams(dense2)
		dense3 = optimizer.UpdateParams(dense3)
		optimizer.PostUpdateParams()

	}

	return
}
