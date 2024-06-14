import pandas as pd
import matplotlib.pyplot as plt

# Path to the parquet file
file_path = '/Users/noahvanrijn/python-repos/master/Thesis/datasets/original_datasets/counselingquestions.parquet'

# Read the parquet file
df = pd.read_parquet(file_path)

# Drop duplicates based on a specific column
df_title = df.drop_duplicates(subset=['questionTitle'], keep=False)
df = df.drop_duplicates(subset=['questionText'], keep=False)

# Get the unique values in a column
distinct_values = df['topic'].unique()

# Get the occurence percentage of each distinct topic
topic_percentage = df['topic'].value_counts(normalize=True) * 100

# Define the topics of interest
topics_of_interest = ['relationships', 'depression', 'intimacy', 'anxiety', 'family-conflict', 'parenting']

# Separate the DataFrame into two parts:
# 1. Rows with topics of interest
df_interest = df[df['topic'].isin(topics_of_interest)]

# 2. Rows with other topics
df_others = df[~df['topic'].isin(topics_of_interest)]

# Apply the filtering condition (at least 75 tokens in 'questionText') to the topics of interest
df_interest_filtered = df_interest[df_interest['questionText'].apply(lambda x: len(x.split()) >= 75)]

# Apply the filtering condition (at least 50 tokens in 'questionText') to the other topics
df_others_filtered = df_others[df_others['questionText'].apply(lambda x: len(x.split()) >= 50)]

# Concatenate the filtered DataFrame with the 'other topics' DataFrame
df = pd.concat([df_interest_filtered, df_others_filtered])

# Calculate target number of occurrences per topic
total_target = 77
unique_topics = df['topic'].nunique()
target_per_topic = total_target // unique_topics
print(target_per_topic)

# Randomly sample from each topic
sampled_dfs = []
for topic in df['topic'].unique():
    topic_df = df[df['topic'] == topic]
    sample_size = min(len(topic_df), target_per_topic)
    sampled_df = topic_df.sample(n=sample_size, random_state=1)
    sampled_dfs.append(sampled_df)

# Concatenate the sampled dataframes
df_sampled = pd.concat(sampled_dfs)

# If we have less than the total target due to small categories, sample more from the larger categories
if len(df_sampled) < total_target:
    remaining = total_target - len(df_sampled)
    additional_samples = df[~df.index.isin(df_sampled.index)].sample(n=remaining, random_state=1)
    df_sampled = pd.concat([df_sampled, additional_samples])

df_sampled = df_sampled[['topic', 'questionText']]

# Save the dataframe as csv
#file_path = '/Users/noahvanrijn/python-repos/master/Thesis/datasets/prepared_datasets/counseling_questions_filtered.csv'
#df_sampled.to_csv(file_path, index=False)

# Total number of occurrences per topic
topic_occurence = df_sampled['topic'].value_counts()

print(df_sampled.head())

print(len(df_sampled))
print(topic_occurence)

# Create a horizontal bar plot
topic_occurence.plot(kind='barh')

# Adding labels and title for clarity
plt.xlabel('Number of questions')
plt.ylabel('Topics')
plt.title('Number of questions for Each Topic')

# Show the plot
plt.show()
