package chp10

import (
	"aiPr/chp3"
	"aiPr/chp4"
	"aiPr/chp5"
	"fmt"
	"gorgonia.org/tensor"
)

func RunDecayFunc11() {
	startingLearningRate := 1.0
	learningRateDecay := 0.1

	for step := 0; step < 20; step++ {
		learningRate := startingLearningRate * (1.0 / (1.0 + learningRateDecay*float64(step)))
		fmt.Println("Learning Rate:", learningRate)
	}
}

func RunUpstreamTestwithDecayFunc12() {

	///////forwarding
	//layer 1
	dense1 := chp3.NewLayerDense(2, 64)
	activation1 := chp4.NewActivationReLU()
	//layer 2
	dense2 := chp3.NewLayerDense(64, 3)
	lossActiovatoin := chp5.NewActivationSoftmaxLoss()
	optimizer := NewOptimizerSGD(1, 1e-3, 0.9)
	var accuracy = float64(0)
	for i := 1; i <= 20000; i++ {

		dense1.Forward(chp3.X)
		activation1.Forward(dense1.Output)
		dense2.Forward(activation1.Output)
		lossActiovatoin.Forward(dense2.Output, chp3.Yval)

		/////// calculate accuracy
		predictions, _ := tensor.Argmax(lossActiovatoin.ActivationSoftMax.Output, 1)
		accuracy = chp5.Accuracy(predictions.Data().([]int), chp3.Yval.Data().([]float64))

		////// backward
		//fmt.Println(lossActiovatoin.ActivationSoftMax.Output, lossActiovatoin.Loss.DInputs)

		lossActiovatoin.Backward(lossActiovatoin.ActivationSoftMax.Output, chp3.Yval)
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
			fmt.Println("acc", accuracy)
			fmt.Println("CurrentLearningRate", optimizer.CurrentLearningRate)
		}

	}
	//fmt.Println("first Layer", dense1.Weights, dense1.Biases)
	//fmt.Println("second Layer", dense2.Weights, dense2.Biases)

	return
}

func RunUpstreamTestwithAdagradFunc13() {

	///////forwarding
	//layer 1
	dense1 := chp3.NewLayerDense(2, 64)
	activation1 := chp4.NewActivationReLU()
	//layer 2
	dense2 := chp3.NewLayerDense(64, 3)
	lossActiovatoin := chp5.NewActivationSoftmaxLoss()
	optimizer := NewOptimizerAdagrad(1, 1e-4, 1e-7)
	var accuracy = float64(0)
	for i := 1; i <= 10000; i++ {

		dense1.Forward(chp3.X)
		activation1.Forward(dense1.Output)
		dense2.Forward(activation1.Output)
		lossActiovatoin.Forward(dense2.Output, chp3.Yval)

		/////// calculate accuracy
		predictions, _ := tensor.Argmax(lossActiovatoin.ActivationSoftMax.Output, 1)
		accuracy = chp5.Accuracy(predictions.Data().([]int), chp3.Yval.Data().([]float64))

		////// backward
		//fmt.Println(lossActiovatoin.ActivationSoftMax.Output, lossActiovatoin.Loss.DInputs)
		lossActiovatoin.Backward(lossActiovatoin.ActivationSoftMax.Output, chp3.Yval)
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
			fmt.Println("acc", accuracy)
			fmt.Println("CurrentLearningRate", optimizer.CurrentLearningRate)
			//fmt.Println(dense1.DWeights.Data())
			//fmt.Println(dense2.DWeights.Data())
		}

	}
	//fmt.Println("first Layer", dense1.Weights, dense1.Biases)
	//fmt.Println("second Layer", dense2.Weights, dense2.Biases)

	return
}

func RunUpstreamTestwithRMSpropFunc14() {

	///////forwarding
	//layer 1
	dense1 := chp3.NewLayerDense(2, 64)
	activation1 := chp4.NewActivationReLU()
	//layer 2
	dense2 := chp3.NewLayerDense(64, 3)
	lossActiovatoin := chp5.NewActivationSoftmaxLoss()
	optimizer := NewOptimizerRMSprop(0.001, 1e-4, 1e-7, 0.9)
	var accuracy = float64(0)
	for i := 1; i <= 10000; i++ {

		dense1.Forward(chp3.X)
		activation1.Forward(dense1.Output)
		dense2.Forward(activation1.Output)
		lossActiovatoin.Forward(dense2.Output, chp3.Yval)

		/////// calculate accuracy
		predictions, _ := tensor.Argmax(lossActiovatoin.ActivationSoftMax.Output, 1)
		accuracy = chp5.Accuracy(predictions.Data().([]int), chp3.Yval.Data().([]float64))

		////// backward
		//fmt.Println(lossActiovatoin.ActivationSoftMax.Output, lossActiovatoin.Loss.DInputs)
		lossActiovatoin.Backward(lossActiovatoin.ActivationSoftMax.Output, chp3.Yval)
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
			fmt.Println("acc", accuracy)
			fmt.Println("CurrentLearningRate", optimizer.CurrentLearningRate)
			//fmt.Println(dense1.DWeights.Data())
			//fmt.Println(dense2.DWeights.Data())
		}

	}
	//fmt.Println("first Layer", dense1.Weights, dense1.Biases)
	//fmt.Println("second Layer", dense2.Weights, dense2.Biases)

	return
}

func RunUpstreamTestwithAdamFunc15() {

	///////forwarding
	//layer 1
	dense1 := chp3.NewLayerDense(2, 64)
	activation1 := chp4.NewActivationReLU()
	//layer 2
	dense2 := chp3.NewLayerDense(64, 3)
	lossActiovatoin := chp5.NewActivationSoftmaxLoss()
	optimizer := NewOptimizerAdam(0.02, 1e-5, 1e-7, 0.9, 0.999)
	var accuracy = float64(0)
	for i := 1; i <= 109; i++ {

		dense1.Forward(chp3.X)
		activation1.Forward(dense1.Output)
		dense2.Forward(activation1.Output)
		lossActiovatoin.Forward(dense2.Output, chp3.Yval)

		/////// calculate accuracy
		predictions, _ := tensor.Argmax(lossActiovatoin.ActivationSoftMax.Output, 1)
		accuracy = chp5.Accuracy(predictions.Data().([]int), chp3.Yval.Data().([]float64))

		////// backward
		//fmt.Println(lossActiovatoin.ActivationSoftMax.Output, lossActiovatoin.Loss.DInputs)
		lossActiovatoin.Backward(lossActiovatoin.ActivationSoftMax.Output, chp3.Yval)
		dense2.Backward(lossActiovatoin.DInputs)
		activation1.Backward(dense2.DInput)
		dense1.Backward(activation1.DInput)
		optimizer.PreUpdateParams()
		dense1 = optimizer.UpdateParams2(dense1)
		dense2 = optimizer.UpdateParams2(dense2)
		optimizer.PostUpdateParams()
		fmt.Println("ITERATION", i, lossActiovatoin.Loss.DataLoss)
		if i%100 == 0 {
			fmt.Println("step", i)
			fmt.Println("loss", lossActiovatoin.Loss.DataLoss)
			fmt.Println("acc", accuracy)
			fmt.Println("CurrentLearningRate", optimizer.CurrentLearningRate)
			//	fmt.Println(dense1.DWeights.Data())
			//	fmt.Println(dense2.DWeights.Data())
		}

	}
	//fmt.Println("first Layer", dense1.Weights, dense1.Biases)
	//fmt.Println("second Layer", dense2.Weights, dense2.Biases)

	return
}
