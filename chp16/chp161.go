package chp16

import (
	"aiPr/chp10"
	"aiPr/chp3"
	"aiPr/chp4"
	"aiPr/chp5"
	"fmt"
	"gorgonia.org/tensor"
)

func RunRegressionFunc1() {
	outputs := []float64{
		1, 2, 3,
		2, 4, 6,
		0, 5, 10,
		11, 12, 13,
		5, 10, 15,
	}
	t := tensor.New(tensor.WithShape(5, 3), tensor.WithBacking(outputs))
	meanOutput := meanAlongLastAxis(TensorToFloat64Slice(t))
	fmt.Println(meanOutput)
}

func TensorToFloat64Slice(t *tensor.Dense) [][]float64 {
	data := make([][]float64, t.Shape()[0])
	for i := 0; i < t.Shape()[0]; i++ {
		data[i] = make([]float64, t.Shape()[1])
		for j := 0; j < t.Shape()[1]; j++ {
			n, _ := t.At(i, j)
			data[i][j] = n.(float64)
		}
	}
	return data
}

func meanAlongLastAxis(outputs [][]float64) []float64 {
	means := make([]float64, len(outputs))
	for i, row := range outputs {
		sum := 0.0
		for _, value := range row {
			sum += value
		}
		fmt.Println(sum, row)
		means[i] = sum / float64(len(row))
	}
	return means
}

func RunRegressionFunc2() {

	///////forwarding
	//layer 1
	dense1 := chp3.NewLayerDense(2, 64, 0, 5e-4, 0, 5e-4)
	activation1 := chp4.NewActivationReLU()

	//layer 2
	dense2 := chp3.NewLayerDense(64, 1)
	activation2 := chp4.NewActivationSigmoid()
	lossFunction := chp5.NewBinaryCrossentropy()
	optimizer := chp10.NewOptimizerAdam(0.001, 5e-7, 1e-7, 0.9, 0.999)
	//optimizer2 := NewOptimizerAdam(0.02, 5e-7, 1e-7, 0.009, 0.000999)
	var accuracy = float64(0)
	for i := 1; i <= 10000; i++ {

		dense1.Forward(chp3.X2Class)
		activation1.Forward(dense1.Output)
		dense2.Forward(activation1.Output)
		activation2.Forward(dense2.Output)
		dataLoss := lossFunction.Forward(activation2.Outputs, chp3.Y2Class)
		//calc regularize loss
		regularizationLoss := lossFunction.Loss.RegularizationLoss(dense1) + lossFunction.Loss.RegularizationLoss(dense2)
		loss := dataLoss + regularizationLoss

		/////// calculate accuracy
		predictions := chp5.Threshold(activation2.Outputs, 0.5)

		ytens := chp3.Y2Class.Clone().(*tensor.Dense)
		accuracy = chp5.AccuracyByTensor(predictions, ytens)

		////// backward
		lossFunction.Backward(activation2.Outputs, chp3.Y2Class)
		activation2.Backward(lossFunction.DInputs)

		dense2.Backward(activation2.DInputs)

		activation1.Backward(dense2.DInput)
		dense1.Backward(activation1.DInput)

		//optimization
		optimizer.PreUpdateParams()
		dense1 = optimizer.UpdateParams(dense1)
		dense2 = optimizer.UpdateParams(dense2)
		optimizer.PostUpdateParams()

		if i%100 == 0 {
			fmt.Println("step", i)
			fmt.Println("loss", dataLoss)
			fmt.Println("regulazrization pen", regularizationLoss)
			fmt.Println("loss with penalty (regulazrization)", loss)
			fmt.Println("acc", accuracy)
			fmt.Println("CurrentLearningRate", optimizer.CurrentLearningRate)

		}

	}

	return
}
