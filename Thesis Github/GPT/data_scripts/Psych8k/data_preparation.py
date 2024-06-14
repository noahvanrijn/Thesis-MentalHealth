import json

# CONVERT JSON DATA TO JSONL FORMAT
file_path = "/Users/noahvanrijn/python-repos/master/Thesis/datasets/original_datasets/psych8k.json"

# Load the JSON data
with open(file_path, "r") as f:
    json_data = json.load(f)

# Function to transform the given JSON data to the desired JSONL format
def convert_to_jsonl(data):
    jsonl_lines = []

    for item in data:
        # Creating the JSONL line for each item
        jsonl_line = {
            "messages": [
                {"role": "system", "content": item["instruction"]},
                {"role": "user", "content": item["input"]},
                {"role": "assistant", "content": item["output"]}
            ]
        }
        jsonl_lines.append(json.dumps(jsonl_line))

    write_path = "/Users/noahvanrijn/python-repos/master/Thesis/datasets/prepared_datasets/psych8k.jsonl"
    # Writing to a JSONL file
    with open(write_path, 'w') as f:
        for line in jsonl_lines:
            f.write(line + '\n')

# Calling the function
convert_to_jsonl(json_data)

print("Conversion complete. The data has been written to 'output.jsonl'.")

