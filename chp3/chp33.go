package chp3

import (
	"fmt"
)

func RunChp32TrainDensLayer() {
	dense1 := NewLayerDense(2, 3)
	dense1.Forward(X)
	//PrintOutput(0, 5, dense1.Output)
	dense2 := NewLayerDense(3, 1)
	dense2.Forward(dense1.Output)
	PrintOutput(0, 5, dense2.Output)
	fmt.Println(dense2.Output)

	return
}
