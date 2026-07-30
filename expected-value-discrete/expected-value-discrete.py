import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x = np.array(x, dtype=float)
    p = np.array(p, dtype=float)

    if(abs(np.sum(p))!=1):
        raise ValueError("Probabilties must sum up to 1")
    return np.sum(x*p)
    pass
