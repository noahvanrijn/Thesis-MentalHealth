import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

file_path = "/Users/noahvanrijn/python-repos/master/Thesis/results/dialogue/clean_dialogue.csv"

# Load the dataframe
data = pd.read_csv(file_path)

# Define the corrected criteria
criteria = [
    "Informativeness", "Guidance", "Empathy ", "Relevance", "Understanding",
    "Exploration ", "Coherence ", "Reflectiveness", "Engagement "
]

# Dictionary to store results
results = {}

# Calculate effect size for each criterion
for criterion in criteria:
    # Extract scores for Fine-tuning and RAG
    fine_tuning_scores = data[criterion]
    rag_scores = data[f"{criterion}.1"]
    
    # Perform the Wilcoxon signed-rank test
    stat, p_value = wilcoxon(fine_tuning_scores, rag_scores)
    
    # Calculate the z-value approximation
    N = len(fine_tuning_scores)
    mean_T = N * (N + 1) / 4
    std_T = np.sqrt(N * (N + 1) * (2 * N + 1) / 24)
    z_value = (stat - mean_T) / std_T
    
    # Calculate the effect size
    r = z_value / np.sqrt(N)
    
    # Store the results
    results[criterion.strip()] = {
        "Statistic": stat,
        "p-value": p_value,
        "z-value": z_value,
        "Effect Size": r
    }

# Convert the results to a DataFrame for display
results_df = pd.DataFrame.from_dict(results, orient='index')

print(results_df)
