import pandas as pd
import matplotlib.pyplot as plt

# Load the provided CSV file
file_path = '/Users/noahvanrijn/python-repos/master/Thesis/results/QA/Final_QA.csv'
df = pd.read_csv(file_path)

# Define the criteria columns for both chatbots
criteria_columns_full = [
    "Informativeness", "Guidance", "Empathy", "Relevance", 
    "Understanding"
]

# Add missing columns to the dataframe for the example consistency
for col in criteria_columns_full:
    if col not in df.columns:
        df[col] = pd.NA

# Calculate the maximum frequency for y-axis limit across all criteria
max_frequency = 80

# Update the plotting function to match the example colors
def create_combined_histograms(df, columns):
    bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]  # Define the bin edges
    fig, axs = plt.subplots(2, 3, figsize=(20, 15))
    fig.suptitle('Comparison of Chatbots', fontsize=20)

    for ax, column in zip(axs.flatten(), columns):
        if column not in df.columns or df[column].dropna().empty:
            ax.set_title(f"{column} (no data)")
            ax.set_ylabel('Frequency')
        else:
            df[df['Chatbot'] == 'CB1'][column].dropna().plot.hist(ax=ax, bins=bins, alpha=0.7, rwidth=0.8, color='blue', label='Fine-tuning')
            df[df['Chatbot'] == 'CB2'][column].dropna().plot.hist(ax=ax, bins=bins, alpha=0.7, rwidth=0.8, color='red', label='RAG')
            ax.set_title(column)
            ax.set_ylabel('Frequency')
            ax.set_ylim(0, max_frequency)  # Set the y-axis limit
            ax.set_xticks([1, 2, 3, 4, 5])  # Set x-ticks to integer values

    # Hide the last subplot (axs[1, 2]) as we only have 5 plots
    fig.delaxes(axs[1, 2])

    # Add x-axis label "Score" to the bottom row
    for ax in axs[1, :]:
        ax.set_xlabel('Score')

    # Create a single legend for all subplots
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', title='Chatbot')

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Create combined histograms for all criteria
create_combined_histograms(df, criteria_columns_full)
