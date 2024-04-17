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

	a := t.New(t.WithShape(1, 3), t.WithBacking([]float32{1, 2, 3}))
	b := t.New(t.WithShape(1, 3), t.WithBacking([]float32{2, 3, 4}))
	b.T()

	c, _ := t.Dot(a, b)
	fmt.Println(c)

}

func RunChp23FNBatch() {
	defer func(timer time.Time) {

		fmt.Println("  chapter batch with dot total took : ", time.Since(timer))
	}(time.Now())

	rawInputs := []float32{
		1, 2, 3, 2.5,
		2, 5, -1, 2,
		-1.5, 2.7, 3.3, -0.8,
	}
	inputs := t.New(t.WithShape(3, 4), t.WithBacking(rawInputs))

	rawWeights := []float32{0.2, 0.8, -0.5, 1,
		0.5, -0.91, 0.26, -0.5,
		-0.26, -0.27, 0.17, 0.87}
	weights := t.New(t.WithShape(3, 4), t.WithBacking(rawWeights))
	weights.T()
	bias := []float64{2, 3, 0.5}

	dot, err := t.Dot(inputs, weights)
	fmt.Println(dot, err)
	output, err := AddBiases(dot, bias)
	fmt.Println(output, err)
	return
}
