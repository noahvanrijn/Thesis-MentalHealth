import numpy as np
from statsmodels.stats.power import TTestIndPower
from scipy.stats import norm

def wilcoxon_signed_rank_sample_size(effect_size, alpha, power):
    """
    Calculate the required sample size for a Wilcoxon signed-rank test using more appropriate methods.

    Parameters:
    - effect_size: The expected effect size (Cohen's d).
    - alpha: The significance level.
    - power: The desired power of the test (1 - beta).

    Returns:
    - The required sample size.
    """
    # Convert the effect size for non-parametric tests
    effect_size_non_parametric = effect_size / np.sqrt(2)
    
    # Calculate the z-scores for alpha and power
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    
    # Sample size formula for Wilcoxon signed-rank test
    n = ((z_alpha + z_beta) / effect_size_non_parametric) ** 2
    
    # Return the sample size, rounded up
    return int(np.ceil(n))

# Parameters
effect_size = 0.5  # Medium effect size for parametric, needs conversion
alpha = 0.05
power = 0.80

# Calculate sample size
sample_size = wilcoxon_signed_rank_sample_size(effect_size, alpha, power)
print(f'Required sample size: {sample_size}')
