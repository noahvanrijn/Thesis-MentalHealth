import pandas as pd
import json

data_path = "/Users/noahvanrijn/python-repos/master/Thesis/datasets/counsel_chat.csv"

# Load the dataset
df = pd.read_csv(data_path)

print(df.head())

# print all columns
print(df.columns)

INSTRUCTION = "If you are a counsellor, please answer the questions based on the description of the patient."

def transform_to_jsonl(df):
    # Initialize a list to store the JSONL lines
    jsonl_lines = []

    # Iterate over the DataFrame rows
    for index, row in df.iterrows():
        # Creating the JSONL line for each item
        jsonl_line = {
            "messages": [
                {"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": row["questionText"]},
                {"role": "assistant", "content": row['answerText']}
            ]
        }
        jsonl_lines.append(json.dumps(jsonl_line))

    write_path = "/Users/noahvanrijn/python-repos/master/Thesis/datasets/prepared_datasets/counsel_chat.jsonl"
    # Writing to a JSONL file
    with open(write_path, 'w') as f:
        for line in jsonl_lines:
            f.write(line + '\n')


def transform_to_json_with_metadata(df):
    # Initialize a list to store the transformed data
    transformed_data = []

    # Iterate over the DataFrame rows
    for index, row in df.iterrows():
        # Combine the metadata into a single string or a dictionary
        # Adjust the formatting according to your requirements
        metadata = {
            'questionTitle': row['questionTitle'],
            'topic': row['topic'],
            'upvotes': row['upvotes']
        }
    
        # Create a dictionary for the current row with the text and metadata
        item = {
            'text': 'Question: ' + row['questionText'] + " , answer: " + row['answerText'],  # Combining question and answer
            'metadata': metadata
        }
    
        # Add the item to the list
        transformed_data.append(item)

    # Print the first transformed item
    print(transformed_data[0])

    # write the path to the file where you want to save the transformed data
    write_path = "/Users/noahvanrijn/python-repos/master/Thesis/GPT-3.5/prepared_data/counsel_chat_transformed.jsonl"

    # Example: Saving the transformed data as a JSON file
    # with open(write_path, 'w') as f:
    #     json.dump(transformed_data, f, ensure_ascii=False, indent=4)

    with open(write_path, 'w') as outfile:
     for entry in transformed_data:
            json.dump(entry, outfile)
            outfile.write('\n')

transform_to_jsonl(df)
transform_to_json_with_metadata(df)