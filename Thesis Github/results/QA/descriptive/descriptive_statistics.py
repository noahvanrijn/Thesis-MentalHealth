import pandas as pd

# Path to the CSV file
file_path = '/Users/noahvanrijn/python-repos/master/Thesis/results/QA/Final_QA.csv'

# Load the dataframe
df = pd.read_csv(file_path)

# Group by 'Chatbot' and calculate descriptive statistics
descriptive_stats = df.groupby('Chatbot').describe().transpose()

print(descriptive_stats)

# Save the descriptive statistics to a new CSV file
descriptive_stats.to_csv('Descriptive_Statistics_chatbots.csv')

# Group by 'Chatbot' and calculate descriptive statistics
descriptive_stats = df.groupby('Psychologist ID').describe().transpose()

print(descriptive_stats)

# Save the descriptive statistics to a new CSV file
descriptive_stats.to_csv('Descriptive_Statistics_psychologists.csv')