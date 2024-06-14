import pandas as pd
import matplotlib.pyplot as plt

# Path to the CSV file
file_path = '/Users/noahvanrijn/python-repos/master/Thesis/results/dialogue/clean_dialogue.csv'

# Load the dataframe
df = pd.read_csv(file_path)

demographic_columns = ['Gender', 'Age', 'Education']

criteria_columns = [
    "Informativeness", "Guidance", "Empathy ", "Relevance", 
    "Understanding", "Exploration ", "Coherence ", "Reflectiveness", "Engagement ",
    "Informativeness.1", "Guidance.1", "Empathy .1", "Relevance.1", 
    "Understanding.1", "Exploration .1", "Coherence .1", "Reflectiveness.1", "Engagement .1"
]

# Separate columns into two groups: chatbot 1 and chatbot 2
chatbot1_columns = [col for col in criteria_columns if not col.endswith('.1')]
chatbot2_columns = [col for col in criteria_columns if col.endswith('.1')]

#--------------------SUMMARY STATISTICS--------------------
# Get the summary statistics of the dataframe
def summary_statistics(df):
    return df.describe()

# Create piechart function
def create_piechart(ax, df, column):
    # Get the value counts of the column
    value_counts = df[column].value_counts()
    
    # Create a pie chart
    wedges, texts, autotexts = ax.pie(value_counts, autopct='%1.1f%%', startangle=90)
    ax.set_title(column)
    
    # Add legend above the plot
    ax.legend(wedges, value_counts.index, title=column, loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=3)
    
    return value_counts

# Calculate the maximum frequency for y-axis limit
max_frequency = 0
for column in chatbot1_columns + chatbot2_columns:
    max_frequency = max(max_frequency, df[column].value_counts().max())

# Create histograms function
def create_histograms(df, columns, axarr, max_y):
    bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]  # Define the bin edges
    for ax, column in zip(axarr, columns):
        if df[column].dropna().empty:
            ax.set_title(f"{column} (no data)")
            ax.set_ylabel('Frequency')
        else:
            df[column].dropna().plot.hist(ax=ax, bins=bins, alpha=0.7, rwidth=0.8)
            ax.set_title(column)
            ax.set_ylabel('Frequency')
            ax.set_ylim(0, max_y)  # Set the y-axis limit
            ax.set_xticks([1, 2, 3, 4, 5])  # Set x-ticks to integer values

#------------------------------------------------------------------------

# Get the summary statistics of the dataframe and save to CSV
summary_stats = summary_statistics(df)
summary_stats.to_csv('summary_statistics.csv')
print(summary_stats)

#---------------------------PIE CHARTS-------------------------------
# Create a figure with 3 subplots arranged in a 1x3 grid
fig, axs = plt.subplots(1, 3, figsize=(20, 8))

# Create a pie chart for each demographic column
for ax, column in zip(axs, demographic_columns):
    create_piechart(ax, df, column)

# Adjust layout to accommodate legends
plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.show()

# Pie chart for the preference column
# Create pie chart for the column "preference"
fig, ax = plt.subplots(figsize=(8, 8))
create_piechart(ax, df, 'Preference')
plt.tight_layout()
plt.show()

# ---------------------------HISTOGRAMS-------------------------------
# Create a figure with subplots arranged in a 3x3 grid for each chatbot
fig, axs = plt.subplots(3, 3, figsize=(20, 15))
fig.suptitle('Chatbot 1', fontsize=20)
create_histograms(df, chatbot1_columns, axs.flatten(), max_frequency)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

fig, axs = plt.subplots(3, 3, figsize=(20, 15))
fig.suptitle('Chatbot 2', fontsize=20)
create_histograms(df, chatbot2_columns, axs.flatten(), max_frequency)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()