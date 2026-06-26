import numpy as np

class GAN:
    def __init__(self, G_W, D_W):
        """
        Initialize GAN with concrete weights.
        """
        self.G_W = np.array(G_W, dtype=float)
        self.D_W = np.array(D_W, dtype=float)

    def generate(self, z):
        """
        Generate fake samples from noise z using tanh(z @ G_W).
        Returns list of lists, rounded to 4 decimals.
        """
        z = np.array(z, dtype=float)
        fake = np.tanh(np.dot(z, self.G_W))
        return np.round(fake, 4).tolist()

    def discriminate(self, x):
        """
        Classify samples using sigmoid(x @ D_W).
        Returns list of lists, rounded to 4 decimals.
        """
        x = np.array(x, dtype=float)

        logits = np.dot(x, self.D_W)
        probs = 1 / (1 + np.exp(-logits))

        return np.round(probs, 4).tolist()

    def train_step(self, real_data, z):
        """
        Compute d_loss and g_loss for one training step.
        Returns dict with "d_loss" and "g_loss", rounded to 4 decimals.
        """
        real_data = np.array(real_data, dtype=float)
        z = np.array(z, dtype=float)

        fake_data = np.tanh(np.dot(z, self.G_W))

        real_prob = 1 / (1 + np.exp(-np.dot(real_data, self.D_W)))
        fake_prob = 1 / (1 + np.exp(-np.dot(fake_data, self.D_W)))

        eps = 1e-8
        real_prob = np.clip(real_prob, eps, 1 - eps)
        fake_prob = np.clip(fake_prob, eps, 1 - eps)

        d_loss = -np.mean(np.log(real_prob) + np.log(1 - fake_prob))
        g_loss = -np.mean(np.log(fake_prob))

        return {
            "d_loss": round(float(d_loss), 4),
            "g_loss": round(float(g_loss), 4)
        }