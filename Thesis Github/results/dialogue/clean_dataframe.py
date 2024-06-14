import pandas as pd
import re

file_path = 'dialogue.csv'

df = pd.read_csv(file_path)

for column in df.columns:
    print(column)

# Rename columns
df.columns = [col.split('\n')[0] for col in df.columns]

for column in df.columns:
    print(column)

# save the dataframe as csv
df.to_csv('dialogue.csv', index=False)

# print the the row of the column "informativeness"
print(df['Informativeness'])

# Define the criteria columns
criteria_columns = [
    "Informativeness", "Guidance", "Empathy ", "Relevance", 
    "Understanding", "Exploration ", "Coherence ", "Reflectiveness", "Engagement ",
    "Informativeness.1", "Guidance.1", "Empathy .1", "Relevance.1", 
    "Understanding.1", "Exploration .1", "Coherence .1", "Reflectiveness.1", "Engagement .1"
]
# Function to extract the numerical value from the criteria columns
def extract_number(text):
    if isinstance(text, str):
        return int(text.split(":")[0])
    return text

# Apply the function to each of the criteria columns
for column in criteria_columns:
    if column in df.columns:
        df[column] = df[column].apply(extract_number)

# print the the row of the column "informativeness"
print(df['Informativeness'])
#print(df['Informativeness.1'])

# Ensure columns are numeric
df[criteria_columns] = df[criteria_columns].apply(pd.to_numeric, errors='coerce')

df = df.rename(columns={
    'Which Mental Health problem do you have to simulate?': 'MentalHealthProblem',
    'What is your gender?': 'Gender',
    'What is your age?': 'Age',
    'Which country are you originally from?': 'Country',
    'What is your highest level of education?': 'Education',
    'If you would have mental health related issues, which chatbot would you prefer?': 'Preference'
})

# Save the updated dataframe to a new CSV file if needed
df.to_csv('clean_dialogue.csv', index=False)