package chp9

import (
	"aiPr/chp3"
	"aiPr/chp4"
	"aiPr/chp5"
	"fmt"
	"gorgonia.org/tensor"
)

func RunBackpropagationLossFunc8() {
	samples := 3
	numClasses := 3
	yTrue := tensor.New(tensor.WithShape(samples, 1), tensor.WithBacking([]float64{0, 1, 2}))
	dinputs := tensor.New(tensor.WithShape(samples, numClasses), tensor.Of(tensor.Float32), tensor.WithBacking([]float64{
		1, 2, 3,
		5, 6, 7,
		9, 10, 11,
	}))
	//it := yTrue.Iterator()
	for i := 0; i < samples; i++ {
		//for _, err := it.Start(); err == nil; _, err = it.Next() {
		valy := yTrue.Get(i)
		fmt.Println(i, valy)
		idx := int(valy.(float64))
		x, _ := dinputs.At(i, idx)
		dinputs.SetAt(x.(float64)-1.0, i, idx)

		//}
	}

	//var dinputsBacking []float64
	//
	//for _, err := it.Start(); err == nil; _, err = it.Next() {
	//	c := it.Coord()
	//	tmp, _ := yTrue.At(c[0])
	//	correctIndex := float64(tmp.(int))
	//	if correctIndex == float64(c[1]) {
	//		tval, _ := dinputs.At(c...)
	//		val := tval.(float64)
	//		dinputsBacking[c[0]] = val - 1
	//	}
	//}
	fmt.Println("Updated dinputs:")
	fmt.Println(dinputs)
	return
}
func RunBackpropagationLossFunc9() {

	softmaxOutputsB := []float64{0.7, 0.1, 0.2,
		0.1, 0.5, 0.4,
		0.02, 0.9, 0.08}
	softmaxOutputs := tensor.New(tensor.WithShape(3, 3), tensor.WithBacking(softmaxOutputsB))
	classTargetsB := []float64{0.0, 1.0, 1.0}
	classTargets := tensor.New(tensor.WithShape(3, 1), tensor.WithBacking(classTargetsB))

	//use one object loss backward then softmax  activation  backward
	softmaxLoss := chp5.NewActivationSoftmaxLoss()
	softmaxLoss.Backward(softmaxOutputs, classTargets)
	dvalues1 := softmaxLoss.DInputs
	fmt.Println("Gradients: combined loss and activation:", dvalues1)

	//use separate loss backward then softmax  activation  backward
	activation := chp4.ActivationSoftMax{
		Output: softmaxOutputs,
	}
	loss := chp5.Loss{}
	loss.Backward(softmaxOutputs, classTargets)
	//fmt.Println(softmaxOutputs, loss.DInputs)
	activation.Backward(softmaxOutputs, loss.DInputs, 1)
	dvalues2 := activation.DInput

	fmt.Println("Gradients: separate loss and activation:", dvalues2)

	return
}

func RunFullTestwithLossFunc10() {

	///////forwarding
	//layer 1
	dense1 := chp3.NewLayerDense(2, 3)
	activation1 := chp4.NewActivationReLU()
	//layer 2
	dense2 := chp3.NewLayerDense(3, 3)
	lossActiovatoin := chp5.NewActivationSoftmaxLoss()

	dense1.Forward(chp3.X)
	activation1.Forward(dense1.Output)
	dense2.Forward(activation1.Output)
	lossActiovatoin.Forward(dense2.Output, chp3.Yval)
	l1 := lossActiovatoin.Loss.DataLoss
	fmt.Println("loss", lossActiovatoin.Loss.DataLoss)
	/////// calculate accuracy
	predictions, _ := tensor.Argmax(lossActiovatoin.ActivationSoftMax.Output, 1)
	accuracy := chp5.Accuracy(predictions.Data().([]int), chp3.Yval.Data().([]float64))
	fmt.Println("acc", accuracy)
	//	panic(1)
	////// backward
	fmt.Println(lossActiovatoin.ActivationSoftMax.Output, lossActiovatoin.Loss.DInputs)
	lossActiovatoin.Backward(lossActiovatoin.ActivationSoftMax.Output, chp3.Yval)
	dense2.Backward(lossActiovatoin.DInputs)
	activation1.Backward(dense2.DInput)

	dense1.Backward(activation1.DInput)

	fmt.Println("dense1.DWeights", dense1.DWeights)
	fmt.Println("dense1.DBiases", dense1.DBiases)
	fmt.Println("dense2.DWeights", dense2.DWeights)
	fmt.Println("dense1.DBiases", dense1.DBiases)

	dense1.Weights, _ = tensor.Mul(dense1.Weights, dense1.DWeights)
	dense1.Biases = dense1.DBiases
	dense2.Weights, _ = tensor.Mul(dense2.Weights, dense2.DWeights)
	dense2.Biases = dense2.DBiases

	fmt.Println("dense1.DWeights", dense1.Weights)
	fmt.Println("dense1.DBiases", dense1.Biases)
	fmt.Println("dense2.DWeights", dense2.Weights)
	fmt.Println("dense1.DBiases", dense1.Biases)

	dense1.Forward(chp3.X)
	activation1.Forward(dense1.Output)
	dense2.Forward(activation1.Output)
	lossActiovatoin.Forward(dense2.Output, chp3.Yval)
	l2 := lossActiovatoin.Loss.DataLoss
	fmt.Println("loss", lossActiovatoin.Loss.DataLoss)
	/////// calculate accuracy
	predictions, _ = tensor.Argmax(lossActiovatoin.ActivationSoftMax.Output, 1)
	accuracy = chp5.Accuracy(predictions.Data().([]int), chp3.Yval.Data().([]float64))
	fmt.Println("accuracy", accuracy)
	fmt.Println("opt", l1-l2)

	return
}
