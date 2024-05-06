import nnfs
import numpy as np
# import matplotlib.pyplot as plt
#from nnfs.datasets import spiral_data
from nnfs.datasets import sine_data

nnfs.init()

#X, y = spiral_data(samples=100, classes=3)
X, y = sine_data()
# plt.scatter(X[:, 0], X[:, 11])
# plt.show()
np.set_printoptions(threshold=np.inf)
print(X)
print("---------------------")
print(y)