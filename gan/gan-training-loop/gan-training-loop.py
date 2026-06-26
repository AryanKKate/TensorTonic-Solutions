import numpy as np

def train_gan_step(real_data, fake_data, D_W):
    """
    Returns: dict with "d_loss" and "g_loss" as float values
    """
    # Your implementation here
    real_prob = 1/(1+np.exp(-np.dot(real_data,D_W)))
    fake_prob = 1/(1+np.exp(-np.dot(fake_data,D_W)))

    epsilon = 1e-8
    real_prob = np.clip(real_prob, epsilon, 1-epsilon)
    fake_prob = np.clip(fake_prob, epsilon, 1-epsilon)

    d_loss = round(-np.mean((np.log(real_prob) + np.log(1-fake_prob))),4)
    g_loss = round(-np.mean(np.log(fake_prob)),4)   

    return{
        "d_loss" : d_loss,
        "g_loss" : g_loss
    }
    
    pass