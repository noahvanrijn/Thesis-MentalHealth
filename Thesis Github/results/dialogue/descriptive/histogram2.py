import pandas as pd
import matplotlib.pyplot as plt

# Load the dataframe
file_path = '/Users/noahvanrijn/python-repos/master/Thesis/results/dialogue/clean_dialogue.csv'
df = pd.read_csv(file_path)

criteria_columns = [
    "Informativeness", "Guidance", "Empathy ", "Relevance", 
    "Understanding", "Exploration ", "Coherence ", "Reflectiveness", "Engagement ",
    "Informativeness.1", "Guidance.1", "Empathy .1", "Relevance.1", 
    "Understanding.1", "Exploration .1", "Coherence .1", "Reflectiveness.1", "Engagement .1"
]

# Separate columns into two groups: chatbot 1 and chatbot 2
chatbot1_columns = [col for col in criteria_columns if not col.endswith('.1')]
chatbot2_columns = [col for col in criteria_columns if col.endswith('.1')]

# Create histograms function
def create_combined_histograms(df, columns1, columns2):
    bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]  # Define the bin edges
    fig, axs = plt.subplots(3, 3, figsize=(20, 15), sharex='col', sharey='row')
    fig.suptitle('Comparison of Chatbots', fontsize=20)

    for idx, (ax, column1, column2) in enumerate(zip(axs.flatten(), columns1, columns2)):
        if df[column1].dropna().empty and df[column2].dropna().empty:
            ax.set_title(f"{column1} and {column2} (no data)")
        else:
            if not df[column1].dropna().empty:
                df[column1].dropna().plot.hist(ax=ax, bins=bins, alpha=0.7, rwidth=0.8, color='blue', label='Chatbot 1')
            if not df[column2].dropna().empty:
                df[column2].dropna().plot.hist(ax=ax, bins=bins, alpha=0.7, rwidth=0.8, color='red', label='Chatbot 2')
            ax.set_title(column1.replace('.1', ''))
            ax.set_ylim(0, max_frequency)  # Set the y-axis limit
            ax.set_xticks([1, 2, 3, 4, 5])  # Set x-ticks to integer values
            if idx % 3 == 0:  # First column in each row
                ax.set_ylabel('Frequency')
                ax.set_yticks(range(0, max_frequency + 2, 2))  # Set y-ticks at intervals of 2
                ax.set_yticklabels(range(0, max_frequency + 2, 2))  # Ensure y-tick labels are displayed
            else:
                ax.set_yticklabels([])

    # Add x-axis label "Score" to the bottom row
    for ax in axs[2, :]:
        ax.set_xlabel('Score')

    # Create a single legend in the top right corner
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Calculate the maximum frequency for y-axis limit
max_frequency = 0
for column in chatbot1_columns + chatbot2_columns:
    max_frequency = max(max_frequency, df[column].value_counts().max())

# Create combined histograms for both chatbots
create_combined_histograms(df, chatbot1_columns, chatbot2_columns)
