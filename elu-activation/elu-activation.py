def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    # Write code here
    n=len(x)
    for i in range(0,n):
        z=x[i]
        if z<=0:
            x[i]=alpha*(math.exp(z)-1)
        
    return x