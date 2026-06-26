import numpy as np

def detect_mode_collapse(generated_samples, threshold=0.1):
    """
    Returns: dict with "diversity_score" (float) and "is_collapsed" (bool)
    """
    # Your implementation here
    div_score = np.mean(np.std(generated_samples,axis=0))
    is_collapsed = div_score < threshold

    return {
        "diversity_score": round(div_score, 4),
        "is_collapsed": is_collapsed
    }
    
    pass