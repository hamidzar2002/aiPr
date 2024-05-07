package chp3

import (
	"aiPr/ml"
	"fmt"
	t "gorgonia.org/tensor"
	"time"
)

func RunChp3InnerLayer() {

	defer func(timer time.Time) {

		fmt.Println(" chapter3 Inner layer total took : ", time.Since(timer))
	}(time.Now())

	rawInputs := []float32{
		1, 2, 3, 2.5,
		2, 5, -1, 2,
		-1.5, 2.7, 3.3, -0.8,
	}
	inputs := t.New(t.WithShape(3, 4), t.WithBacking(rawInputs))

	//layer 1
	rawWeights1 := []float32{0.2, 0.8, -0.5, 1,
		0.5, -0.91, 0.26, -0.5,
		-0.26, -0.27, 0.17, 0.87}
	weights1 := t.New(t.WithShape(3, 4), t.WithBacking(rawWeights1))
	weights1.T()
	bias1 := []float64{2, 3, 0.5}

	//layer2
	rawWeights2 := []float32{
		0.1, -0.14, 0.5,
		-0.5, 0.12, -0.33,
		-0.44, 0.73, -0.13,
	}
	weights2 := t.New(t.WithShape(3, 3), t.WithBacking(rawWeights2))
	weights2.T()
	bias2 := []float64{-1, 2, -0.5}

	//forward  pass
	dot1, _ := t.Dot(inputs, weights1)
	output1, _ := ml.AddBiases(dot1, bias1)
	fmt.Println("output1 ::>", output1)

	dot2, _ := t.Dot(output1, weights2)
	output2, _ := ml.AddBiases(dot2, bias2)
	fmt.Println("output2 ::>", output2)

	return
}
