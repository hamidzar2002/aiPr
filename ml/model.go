package ml

import (
	"fmt"
	"gorgonia.org/tensor"
	"reflect"
)

type Model struct {
	Layers     []LayerStruct
	Loss       Los
	Optimizer  Optimizer
	LayerInput *LayerInput
}

func (m *Model) Add(layer Layer) {
	var l = LayerStruct{
		layer: layer,
	}
	m.Layers = append(m.Layers, l)
}

func (m *Model) Set(loss Los, opt Optimizer) {
	m.Optimizer = opt
	m.Loss = loss
}

func (m *Model) Train(X, y tensor.Tensor, epoch, printEvery int) {

	for i := 0; i <= epoch; i++ {

		output := m.Forward(X)

		if i%100 == 0 {
			fmt.Println("step", i)
			fmt.Println("output", output)
			//fmt.Println("loss", dataLoss)
			//fmt.Println("regulazrization pen", regularizationLoss)
			//fmt.Println("loss with penalty (regulazrization)", loss)
			//fmt.Println("acc", accuracy)
			//fmt.Println("CurrentLearningRate", optimizer.CurrentLearningRate)

		}
	}

	return
}

func (m *Model) Finalize() {

	m.LayerInput = new(LayerInput)
	layerCount := len(m.Layers)
	for i := 0; i < layerCount; i++ {
		// If it's the first layer,
		// the previous layer object is the input layer
		if i == 0 {
			m.Layers[i].prev = m.LayerInput
			m.Layers[i].next = m.Layers[i+1]
		} else if i < layerCount-1 {
			// All layers except for the first and the last
			m.Layers[i].prev = m.Layers[i-1]
			m.Layers[i].next = m.Layers[i+1]
		} else {
			// The last layer - the next object is the loss
			m.Layers[i].prev = m.Layers[i-1]
			m.Layers[i].next = m.Loss
		}
	}
}
func (m *Model) Forward(X tensor.Tensor) *tensor.Tensor {
	m.LayerInput.Forward(X)

	var l Layer
	for _, ls := range m.Layers {
		currentLayer := ls.layer
		prevLayer := ls.prev

		t := reflect.TypeOf(currentLayer)
		if t.String() == "LayerDense" {
			currentLayer = ls.layer.(LayerDense)
		} else if t.String() == "Layer" {
			currentLayer = ls.layer.(Layer)
		} else if t.String() == "Los" {
			currentLayer = ls.layer.(Los)
		}

		t = reflect.TypeOf(prevLayer)
		if t.String() == "LayerDense" {
			prevLayer = ls.layer.(LayerDense)
		} else if t.String() == "Layer" {
			prevLayer = ls.layer.(Layer)
		} else if t.String() == "Los" {
			prevLayer = ls.layer.(Los)
		}
		//	prevLayer.()

		//currentLayer.Forward(*prevLayer.GetOutput())
		//l = currentLayer

	}

	return l.GetOutput()
}
