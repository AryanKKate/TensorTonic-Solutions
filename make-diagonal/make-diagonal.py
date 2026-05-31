import numpy as np

def make_diagonal(v):
    """
    Returns: (n, n) NumPy array with v on the main diagonal
    """
    # Write code here
    n=len(v)
    v=np.array(v)
    arr=[]

    for i in range(n):
        temp=[]
        for j in range(n):
            if i==j:
                temp.append(v[i])
            else:
                temp.append(0)
        arr.append(temp)

    return np.array(arr)
    pass
