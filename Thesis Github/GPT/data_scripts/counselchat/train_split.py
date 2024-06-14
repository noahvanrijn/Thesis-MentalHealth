import json

# THIS PART IS USED FOR TESTING ON SMALL DATASETS
# ------------------------LOAD THE DATA AND SPLIT INTO TRAINING AND VALIDATION SETS------------------------
file_name = "/Users/noahvanrijn/python-repos/master/Thesis/datasets/prepared_datasets/counsel_chat.jsonl"

# Function to load the JSONL file
def load_jsonl(filename):
    # Initialize an empty list to store the data
    data = []
    # Open the file and read each line
    with open(filename, 'r') as file:
        for line in file:
            # Parse the JSON data and append it to the list
            data.append(json.loads(line))
    return data

# Load the JSONL file
data = load_jsonl(file_name)

# Split the data into training and validation sets
train_percentage = 0.8
val_percentage = 0.1 + train_percentage

train_data = data[:int(train_percentage*len(data))]
validation_data = data[int(train_percentage*len(data)):int(val_percentage*len(data))]
test_data = data[int(val_percentage*len(data)):]

print(validation_data)

# Function to write data to a JSONL file
def write_jsonl(data, filename):
    with open(filename, 'w') as file:
        for item in data:
            json_line = json.dumps(item)  # Convert dict to JSON string
            file.write(json_line + '\n')  # Write JSON string as line in file

# Save the training and validation data to files
write_jsonl(train_data, '/Users/noahvanrijn/python-repos/master/Thesis/datasets/prepared_datasets/counselchat_train.jsonl')
write_jsonl(validation_data, '/Users/noahvanrijn/python-repos/master/Thesis/datasets/prepared_datasets/counselchat_val.jsonl')

# ------------------------END OF TESTING ON SMALL DATASETS------------------------
