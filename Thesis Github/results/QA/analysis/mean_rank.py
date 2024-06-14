import pandas as pd
import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = '/Users/noahvanrijn/python-repos/master/Thesis/results/QA/Final_QA.csv'
data = pd.read_csv(file_path)

# Convert categorical variables to factors
data['Chatbot'] = data['Chatbot'].astype('category')
data['Psychologist_ID'] = data['Psychologist_ID'].astype('category')

# Function to rank the data and calculate mean ranks
def calculate_mean_ranks(data, criterion):
    # Rank transform the criterion
    data[f'{criterion}_rank'] = rankdata(data[criterion], method='average')
    
    # Calculate mean ranks
    mean_ranks = data.groupby('Chatbot')[f'{criterion}_rank'].mean().reset_index()
    mean_ranks['Chatbot'] = mean_ranks['Chatbot'].replace({'CB1': 'RAG', 'CB2': 'Fine-tuning'})
    
    return mean_ranks

# Criteria to analyze
criteria = ['Informativeness', 'Guidance', 'Empathy', 'Relevance', 'Understanding']

# Perform rank transformation and calculate mean ranks for each criterion
mean_ranks_list = []
plots_data = {}

for criterion in criteria:
    mean_ranks = calculate_mean_ranks(data, criterion)
    mean_ranks_list.append(mean_ranks)
    plots_data[criterion] = mean_ranks

# Determine the y-axis limit
max_y = max([mean_ranks[f'{criterion}_rank'].max() for mean_ranks, criterion in zip(mean_ranks_list, criteria)])

# Create the subplot figure
fig, axes = plt.subplots(nrows=1, ncols=len(criteria), figsize=(20, 5), sharey=True)

# Custom colors for the bars
colors = {'RAG': 'red', 'Fine-tuning': 'blue'}

for ax, (criterion, mean_ranks) in zip(axes, plots_data.items()):
    sns.barplot(x='Chatbot', y=f'{criterion}_rank', data=mean_ranks, ax=ax, palette=colors)
    ax.set_title(f'{criterion}')
    ax.set_ylim(0, max_y)
    ax.set_xlabel('' if ax != axes[-1] else 'Chatbot')
    ax.set_ylabel('Mean Rank' if ax == axes[0] else '')

# Adjust the layout
plt.tight_layout()
plt.show()
