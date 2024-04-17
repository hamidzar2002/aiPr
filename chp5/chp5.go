package chp5

import (
	"fmt"
	"gorgonia.org/tensor"
	"math"
)

func RunLossFunc1() {
	//softmaxOutput := []float64{0.7, 0.1, 0.2}
	//targetOutput := []float64{1, 0, 0}
	//los := -(math.Log(softmaxOutput[0])*targetOutput[0] +
	//	math.Log(softmaxOutput[1])*targetOutput[1] +
	//	math.Log(softmaxOutput[2])*targetOutput[2])
	// /*or it can be */ los := -(math.Log(softmaxOutput[0])
	//fmt.Println(los)
	softmaxOutput := [][]float64{
		{0.7, 0.1, 0.2},
		{0.1, 0.5, 0.4},
		{0.02, 0.9, 0.08},
	}
	targetOutput := []float64{0, 1, 1}
	losses := make([]float64, len(targetOutput))
	for i, preds := range softmaxOutput {
		//fmt.Println(preds, int(targetOutput[i]), preds[int(targetOutput[i])])
		losses[i] = -(math.Log(preds[int(targetOutput[i])]))
	}

	fmt.Println(losses)
	return
}
func RunLossFunc2() {

	softmaxOutputBacking := []float64{
		0.7, 0.1, 0.2,
		0.1, 0.5, 0.4,
		0.02, 0.9, 0.08,
	}
	softmaxOutput := tensor.New(tensor.WithShape(3, 3), tensor.WithBacking(softmaxOutputBacking))

	predictedValsBacking := make([]float64, softmaxOutput.Shape()[0])

	it := softmaxOutput.Iterator()
	targetOutput := []float64{0, 1, 1}

	for _, err := it.Start(); err == nil; _, err = it.Next() {
		//val, _ := softmaxOutput.At(it.Coord()...)
		//fmt.Println(i, it.Coord(), val)
		c := it.Coord()
		if int(targetOutput[c[0]]) == c[1] {
			//	predictedValsBacking[]
			tval, _ := softmaxOutput.At(c...)
			val := tval.(float64)
			predictedValsBacking[c[0]] = val
		}
	}

	predictedVals := tensor.New(tensor.WithBacking(predictedValsBacking))
	losses, _ := tensor.Log(predictedVals)
	losses, _ = tensor.Mul(losses, float64(-1))
	fmt.Println(losses)
	sum, _ := tensor.Sum(losses)
	avg, _ := tensor.Div(sum, float64(losses.Size()))
	fmt.Println(sum, avg)
	return
}
func RunLossFunc3() {

	softmaxOutputBacking := []float64{
		0.7, 0.1, 0.2,
		0.1, 0.5, 0.4,
		0.02, 0.9, 0.08,
	}
	softmaxOutput := tensor.New(tensor.WithShape(3, 3), tensor.WithBacking(softmaxOutputBacking))

	predictedValsBacking := make([]float64, softmaxOutput.Shape()[0])

	it := softmaxOutput.Iterator()
	//classTargetsBacking := []float64{0, 1, 1}
	classTargetsBacking := []float64{
		1, 0, 0,
		0, 1, 0,
		0, 1, 0,
	}

	classTargets := tensor.New(
		tensor.WithBacking(classTargetsBacking),
		tensor.WithShape(3, 3),
	)

	//if reflect.TypeOf(targetOutput[0]) == reflect.TypeOf(0.0) {
	//
	//}

	if classTargets.Dims() == 1 {
		for _, err := it.Start(); err == nil; _, err = it.Next() {
			//val, _ := softmaxOutput.At(it.Coord()...)
			//fmt.Println(i, it.Coord(), val)
			c := it.Coord()
			correctIndex, _ := classTargets.At(c[0])
			if correctIndex == float64(c[1]) {
				//	predictedValsBacking[]
				tval, _ := softmaxOutput.At(c...)
				val := tval.(float64)
				predictedValsBacking[c[0]] = val
			}
		}
	} else if classTargets.Dims() == 2 {
		argMax, _ := tensor.Argmax(classTargets, 1)
		//val, _ := softmaxOutput.At(it.Coord()...)
		//fmt.Println(i, it.Coord(), val)
		for _, err := it.Start(); err == nil; _, err = it.Next() {
			c := it.Coord()

			correctIndex, _ := argMax.At(c[0])
			if correctIndex == int(c[1]) {
				//	predictedValsBacking[]
				tval, _ := softmaxOutput.At(c...)
				val := tval.(float64)
				predictedValsBacking[c[0]] = val
			}
		}
	}
	predictedVals := tensor.New(tensor.WithBacking(predictedValsBacking))
	losses, _ := tensor.Log(predictedVals)
	losses, _ = tensor.Mul(losses, float64(-1))
	fmt.Println(losses)
	sum, _ := tensor.Sum(losses)
	avg, _ := tensor.Div(sum, float64(losses.Size()))
	fmt.Println(sum, avg)
	return
}
