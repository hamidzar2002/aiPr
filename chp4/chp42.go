package chp4

import (
	chp3 "aiPr/chp3"
	"fmt"
	"math"
	"time"
)

func RunSoftmax1() {
	defer func(timer time.Time) {

		fmt.Println(" chapter 4 softmax 1 total took : ", time.Since(timer))
	}(time.Now())

	dense1 := chp3.NewLayerDense(2, 3)
	activation1 := NewActivationReLU()
	_, _ = dense1, activation1
	fmt.Println()
	//dense1.Forward(chp3.X)
	//activation1.Forward(dense1.Output)
	//fmt.Println(activation1.Output)

	// softmax formula
	outputs := []float64{4.8, 1.21, 2.385}
	expVals := make([]float64, len(outputs))
	normBase := float64(0)
	for i, val := range outputs {
		expVals[i] = math.Pow(math.E, val)
		normBase += expVals[i]
	}
	normVals := make([]float64, len(expVals))
	sum := float64(0)
	for i, val := range expVals {
		normVals[i] = val / normBase
		sum += normVals[i]
	}
	fmt.Println(normVals, sum)
}
func RunSoftmax2() {
	defer func(timer time.Time) {

		fmt.Println(" chapter 4 softmax 2 total took : ", time.Since(timer))
	}(time.Now())

	dense1 := chp3.NewLayerDense(2, 3)
	activation1 := NewActivationReLU()
	_, _ = dense1, activation1
	//dense1.Forward(chp3.X)
	//activation1.Forward(dense1.Output)
	//fmt.Println(activation1.Output)

	// softmax formula 2 dimension
	outputs := [][]float64{
		{4.8, 1.21, 2.385},
		{8.9, -1.81, 0.2},
		{1.41, 1.051, 0.026},
	}

	// subtract to avoid overflow

	for i, vals := range outputs {
		// getting the max
		max := vals[0]
		for _, val := range vals {
			if val > max {
				max = val
			}
		}

		for j, val := range vals {
			outputs[i][j] = val - max
		}
	}
	//fmt.Println(outputs)

	// exponential vals to remove negatives

	expVals := make([][]float64, len(outputs))
	normBase := make([]float64, len(outputs))

	for i, output := range outputs {
		expVals[i] = make([]float64, len(output))
		for j, val := range output {
			expVals[i][j] = math.Pow(math.E, val)
			normBase[i] += expVals[i][j]
		}
	}

	//normalize to get percentage confidence
	sums := make([]float64, len(expVals))
	normVals := make([][]float64, len(expVals))
	for i, output := range expVals {
		normVals[i] = make([]float64, len(output))
		for j, val := range output {
			normVals[i][j] = val / normBase[i]
			sums[i] += normVals[i][j]
		}
	}

	fmt.Println(expVals, normBase)
	fmt.Println(normVals, sums)

}
func RunSoftmax3() {
	defer func(timer time.Time) {

		fmt.Println(" chapter 4 softmax 3 total took : ", time.Since(timer))
	}(time.Now())

	//layer 1
	dense1 := chp3.NewLayerDense(2, 3)
	activation1 := NewActivationReLU()
	//layer 2
	dense2 := chp3.NewLayerDense(3, 3)
	activation2 := NewActivationSoftMax()

	dense1.Forward(chp3.X)
	activation1.Forward(dense1.Output)

	dense2.Forward(activation1.Output)
	activation2.Forward(dense2.Output, 1)

	fmt.Println(activation2.Output)

}
