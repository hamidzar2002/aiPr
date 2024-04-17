package chp4

import (
	chp3 "aiPr/chp3"
	"fmt"
	"gorgonia.org/tensor"
	"math"
)

func RunReLUfunc1() {

	inputsBacking := []float64{0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100}
	output := make([]float64, len(inputsBacking))
	for i, val := range inputsBacking {
		output[i] = math.Max(0, val)
	}
	fmt.Println(output)

	input := tensor.New(tensor.WithBacking(inputsBacking))
	for _, x := range input.Float64s() {
		fmt.Println(math.Max(0, x))
	}
	return

}

func RunReLUfuncDense() {
	dense1 := chp3.NewLayerDense(2, 3)
	activation1 := NewActivationReLU()

	dense1.Forward(chp3.X)
	activation1.Forward(dense1.Output)
	fmt.Println(activation1.Output)

}
