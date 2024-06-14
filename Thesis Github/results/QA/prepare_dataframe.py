import pandas as pd
import re

file_path = 'QA.csv'
sequence_path = 'sequence.csv'

df = pd.read_csv(file_path)
sequence = pd.read_csv(sequence_path)

for column in df.columns:
    print(column)

# Rename columns
df.columns = [col.split('\n')[0] for col in df.columns]

for column in df.columns:
    print(column)

# Create a new DataFrame to store the transformed data
transformed_data = pd.DataFrame()

# Extract psychologist IDs (assuming they are unique timestamps for simplicity)
psychologist_ids = df['Tijdstempel']

# Number of questions based on the number of sets of criteria columns
num_questions = (df.shape[1] - 1) // 10

# Loop through each question and extract the relevant columns
question_ids = [f'Q{i+1}' for i in range(num_questions)]
for i in range(num_questions):
    start_col = i * 10 + 1
    end_col = start_col + 10
    question_data = df.iloc[:, start_col:end_col]
    question_data.columns = [
        'Informativeness_1', 'Guidance_1', 'Empathy_1', 'Relevance_1', 'Understanding_1',
        'Informativeness_2', 'Guidance_2', 'Empathy_2', 'Relevance_2', 'Understanding_2'
    ]
    question_data.insert(0, 'Psychologist ID', psychologist_ids)
    question_data.insert(0, 'Question ID', question_ids[i])
    transformed_data = pd.concat([transformed_data, question_data], ignore_index=True)

# Display the first few rows of the transformed data
print(transformed_data.head())

# Create an empty DataFrame for the final transformed data
final_transformed_data = pd.DataFrame()

# Iterate through each question in the sequence and split the corresponding rows in qa_transformed
for idx, row in sequence.iterrows():
    question_id = f'Q{row["Question"]}'
    cb1_first = row["Form tune"] == 1
    
    cb1_cols = ['Informativeness_1', 'Guidance_1', 'Empathy_1', 'Relevance_1', 'Understanding_1']
    cb2_cols = ['Informativeness_2', 'Guidance_2', 'Empathy_2', 'Relevance_2', 'Understanding_2']
    
    if not cb1_first:
        cb1_cols, cb2_cols = cb2_cols, cb1_cols
    
    # Get the corresponding rows in qa_transformed
    question_rows = transformed_data[transformed_data['Question ID'] == question_id]
    
    # Split each row into two rows, one for each chatbot
    for _, question_row in question_rows.iterrows():
        cb1_row = {
            'Question ID': question_id,
            'Chatbot': 'CB1',
            'Psychologist ID': question_row['Psychologist ID'],
            'Informativeness': question_row[cb1_cols[0]],
            'Guidance': question_row[cb1_cols[1]],
            'Empathy': question_row[cb1_cols[2]],
            'Relevance': question_row[cb1_cols[3]],
            'Understanding': question_row[cb1_cols[4]]
        }
        
        cb2_row = {
            'Question ID': question_id,
            'Chatbot': 'CB2',
            'Psychologist ID': question_row['Psychologist ID'],
            'Informativeness': question_row[cb2_cols[0]],
            'Guidance': question_row[cb2_cols[1]],
            'Empathy': question_row[cb2_cols[2]],
            'Relevance': question_row[cb2_cols[3]],
            'Understanding': question_row[cb2_cols[4]]
        }
        
        final_transformed_data = pd.concat([final_transformed_data, pd.DataFrame([cb1_row])], ignore_index=True)
        final_transformed_data = pd.concat([final_transformed_data, pd.DataFrame([cb2_row])], ignore_index=True)

# Save the final transformed data to a new CSV file
final_transformed_data.to_csv('Final_QA.csv', index=False)

# Display the first few rows of the final transformed data
print(final_transformed_data.head())

