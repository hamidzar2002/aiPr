package chp2

import (
	"fmt"
	t "gorgonia.org/tensor"
	"time"
)

func RunChp22FN1() {
	defer func(timer time.Time) {

		fmt.Println("total took : ", time.Since(timer))
	}(time.Now())
	fmt.Println("start chapter 2 tensor 1")
	l := [][][]int32{
		{{1, 5, 6, 2},
			{3, 2, 1, 3}},
		{{5, 2, 1, 2},
			{6, 4, 8, 4}},
		{{2, 8, 5, 3},
			{1, 1, 9, 3}}}

	a := []int32{1, 2, 3}
	b := []int32{4, 5, 6}

	c := a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
	fmt.Println(l, c)

	return
}

func RunChp22FNSingleDot() {
	defer func(timer time.Time) {

		fmt.Println("  chapter 2 single neurons with dot total took : ", time.Since(timer))
	}(time.Now())
	inputs := t.New(t.WithBacking([]float32{1, 2, 3, 2.5}))
	weights := t.New(t.WithBacking([]float32{0.2, 0.8, -0.5, 1}))
	bias := t.New(t.WithBacking([]float32{2}))

	dot, _ := t.Dot(weights, inputs)

	output, _ := t.Add(dot, bias)
	fmt.Println(output)
	return
}
func RunChp22FNMultiDot() {
	defer func(timer time.Time) {

		fmt.Println("  chapter 2 Multi neurons with dot total took : ", time.Since(timer))
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
