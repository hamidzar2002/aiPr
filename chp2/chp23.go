package chp2

import (
	"fmt"
	t "gorgonia.org/tensor"
	"time"
)

func RunChp23FNTrans() {
	var inputs t.Tensor = t.New(t.WithShape(1, 4), t.WithBacking([]float32{1.0, 2.0, 3.0, 2.5}))
	fmt.Println(inputs, inputs.Shape())
	inputs.T()
	fmt.Println(inputs, inputs.Shape())
}

func RunChp23FNBatch() {
	defer func(timer time.Time) {

		fmt.Println("  chapter batch with dot total took : ", time.Since(timer))
	}(time.Now())
	inputs := t.New(t.WithBacking([]float32{1, 2, 3, 2.5}))

	rawWeights := []float32{0.2, 0.8, -0.5, 1,
		0.5, -0.91, 0.26, -0.5,
		-0.26, -0.27, 0.17, 0.87}
	weights := t.New(t.WithShape(3, 4), t.WithBacking(rawWeights))
	bias := t.New(t.WithBacking([]float32{2, 3, 0.5}))

	dot, _ := t.Dot(weights, inputs)
	//fmt.Println(dot, err)
	output, _ := t.Add(dot, bias)
	fmt.Println(output)
	return
}
