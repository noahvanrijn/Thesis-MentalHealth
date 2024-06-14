import matplotlib.pyplot as plt
import pandas as pd 

file_path = '/Users/noahvanrijn/python-repos/master/Thesis/results/QA/Final_QA.csv'
df = pd.read_csv(file_path)

# Extract unique psychologist IDs
psychologist_ids = df['Psychologist ID'].unique()

# Create a figure for each psychologist ID with custom labels
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 20), sharex=True)

# Custom labels for the figures
labels = ['Psychologist 1', 'Psychologist 2', 'Psychologist 3']

# Define the criteria
criteria = ['Informativeness', 'Guidance', 'Empathy', 'Relevance', 'Understanding']

# Plot boxplots for each psychologist ID with custom labels
for i, (psych_id, label) in enumerate(zip(psychologist_ids, labels)):
    ax = axes[i]
    subset = df[df['Psychologist ID'] == psych_id]
    subset.boxplot(column=criteria, ax=ax)
    ax.set_title(label)
    ax.set_ylim(0, 6)  # Assuming the scale is from 0 to 5
    ax.set_ylabel('Score')
    ax.grid(False)

plt.tight_layout()
plt.show()

