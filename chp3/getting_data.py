import nnfs
# import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(samples=100, classes=3)
# plt.scatter(X[:, 0], X[:, 11])
# plt.show()

print(X)
print("---------------------")
print(y)