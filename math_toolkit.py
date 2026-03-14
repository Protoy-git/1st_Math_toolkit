import numpy as np

A = np.array([[2, 1], [5, 3]])
print("Matrix:\n", A)
print("Inverse:\n", np.linalg.inv(A))
print("Determinant:", np.linalg.det(A))
print("Transpose:\n", A.T)

data = [4, 8, 6, 5, 3, 2, 8, 9, 2, 5]
print("\nMean:", np.mean(data))
print("Median:", np.median(data))
print("Std Dev:", np.std(data))
