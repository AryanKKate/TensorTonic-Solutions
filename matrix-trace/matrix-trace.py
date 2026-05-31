import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    # Write code here
    sum=0
    n=len(A)
         
    for i in range(0,n):
          sum+=A[i][i]

    return sum
    pass
