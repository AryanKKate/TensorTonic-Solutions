import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.array(A)

    rows = len(A)
    cols = len(A[0])

    X = []

    for i in range(cols):
        temp = []
        for j in range(rows):
            temp.append(A[j][i])
        X.append(temp)

    return np.array(X)