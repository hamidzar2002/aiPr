package chp2

import (
	"fmt"
	"time"
)

func RunChp2FN1() {
	fmt.Println("start chapter 2 first neurons")
	inputs := []float32{1, 2, 3}
	weights := []float32{0.2, 0.8, -0.5}
	bias := float32(2)

	outputs := (inputs[0]*weights[0] +
		inputs[1]*weights[1] +
		inputs[2]*weights[2] + bias)
	fmt.Println(outputs)
	return
}
func RunChp2FN2() {
	defer func(timer time.Time) {

		fmt.Println("total took : ", time.Since(timer))
	}(time.Now())
	fmt.Println("start chapter 2 second neurons")
	inputs := []float32{1, 2, 3, 2.5}
	weights1 := []float32{0.2, 0.8, -0.5, 1}
	weights2 := []float32{0.5, -0.91, 0.26, -0.5}
	weights3 := []float32{-0.26, -0.27, 0.17, 0.87}

	bias1 := float32(2)
	bias2 := float32(3)
	bias3 := float32(0.5)
	outputs := make([]float32, 3)

	outputs[0] = (inputs[0]*weights1[0] +
		inputs[1]*weights1[1] +
		inputs[2]*weights1[2] +
		inputs[3]*weights1[3] + bias1)

	outputs[1] = (inputs[0]*weights2[0] +
		inputs[1]*weights2[1] +
		inputs[2]*weights2[2] +
		inputs[3]*weights2[3] + bias2)

	outputs[2] = (inputs[0]*weights3[0] +
		inputs[1]*weights3[1] +
		inputs[2]*weights3[2] +
		inputs[3]*weights3[3] + bias3)

	fmt.Println(outputs)
	return
}
func RunChp2FN3() {
	defer func(timer time.Time) {

		fmt.Println("total took : ", time.Since(timer))
	}(time.Now())
	fmt.Println("start chapter 2 second neurons")
	inputs := []float32{1, 2, 3, 2.5}
	weights := [][]float32{{0.2, 0.8, -0.5, 1}, {0.5, -0.91, 0.26, -0.5}, {-0.26, -0.27, 0.17, 0.87}}

	bias := []float32{2, 3, 0.5}

	neuronOutputs := make([]float32, 3)

	for i := range weights {
		neuronWeight := weights[i]
		neuronBias := bias[i]

		output := float32(0)
		for j := range inputs {
			output += neuronWeight[j] * inputs[j]
		}
		output += neuronBias
		neuronOutputs[i] = output

	}

	fmt.Println(neuronOutputs)
	return
}
