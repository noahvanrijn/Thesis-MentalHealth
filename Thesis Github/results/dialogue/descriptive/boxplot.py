import pandas as pd
import matplotlib.pyplot as plt

# Load the dataframe
file_path = '/Users/noahvanrijn/python-repos/master/Thesis/results/dialogue/clean_dialogue.csv'
df = pd.read_csv(file_path)

# Columns for each chatbot
criteria_columns = [
    "Informativeness", "Guidance", "Empathy ", "Relevance", 
    "Understanding", "Exploration ", "Coherence ", "Reflectiveness", "Engagement ",
    "Informativeness.1", "Guidance.1", "Empathy .1", "Relevance.1", 
    "Understanding.1", "Exploration .1", "Coherence .1", "Reflectiveness.1", "Engagement .1"
]

chatbot1_columns = [col for col in criteria_columns if not col.endswith('.1')]
chatbot2_columns = [col for col in criteria_columns if col.endswith('.1')]

# Figure for Chatbot 1
plt.figure(figsize=(12, 6))
plt.boxplot([df[col].dropna() for col in chatbot1_columns], labels=chatbot1_columns)
plt.title('Distributions of Human Evaluation Scores for Chatbot 1')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Figure for Chatbot 2
plt.figure(figsize=(12, 6))
plt.boxplot([df[col].dropna() for col in chatbot2_columns], labels=chatbot2_columns)
plt.title('Distributions of Human Evaluation Scores for Chatbot 2')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
