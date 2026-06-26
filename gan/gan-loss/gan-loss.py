import numpy as np

def discriminator_loss(real_probs, fake_probs):
    """Compute discriminator loss using binary cross-entropy.
    Returns: Loss value rounded to 4 decimals."""

    n = len(real_probs)
    epsilon = 1e-8
    
    fake_probs = np.clip(fake_probs, epsilon, 1 - epsilon)
    real_probs = np.clip(real_probs, epsilon, 1 - epsilon)
    term=0
    
    for i in range (0,n):
        term+=(np.log(real_probs[i])+np.log(1-fake_probs[i]))
        
    return round(-1*term/n,4)
        
    pass

def generator_loss(fake_probs):
    """Compute non-saturating generator loss.
    Returns: Loss value rounded to 4 decimals."""

    n = len(fake_probs)
    epsilon = 1e-8
    
    fake_probs = np.clip(fake_probs, epsilon, 1 - epsilon)
    term=0
    
    for i in range (0,n):
        term+=np.log(fake_probs[i])
        
    return round(-1*term/n,4)    
    pass