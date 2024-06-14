import json
import re

# Define a function to remove markdown links
def remove_markdown_links(text):
    # This regex will match markdown links and replace them with just the text part
    return re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

# Load the JSON file
json_file_path = 'mind_dataset.json'

with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Process each document to remove links
for doc in data:
    doc['page_content'] = remove_markdown_links(doc['page_content'])


# Here, we save it to a new JSON file
modified_json_file_path = 'mind_dataset_cleaned.json'

with open(modified_json_file_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, indent=4)


