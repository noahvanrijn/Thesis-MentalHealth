from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

import json
import os

# Load the environment variables
load_dotenv()

# Load the JSON file
json_file_path = '/Users/noahvanrijn/python-repos/master/Thesis/datasets/prepared_datasets/mind_dataset_cleaned.json'

with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Define the headers on which to split the documents
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

# Initialize the MarkdownHeaderTextSplitter
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, strip_headers=True
)

# Split the documents based on the headers
chunks = []
for doc in data:
    splits = markdown_splitter.split_text(doc['page_content'])
    chunks.extend(splits)

# Print the number of documents after splitting
print(f'Number of documents after splitting: {len(chunks)}')

for doc in chunks:
    print(doc.page_content)
    print('\n')
    print('-----------------------------------')


# Initialize the OpenAIEmbeddings class
embedding_model = OpenAIEmbeddings()

# Set up vector database
db = FAISS.from_documents(
    chunks, embedding_model, distance_strategy=DistanceStrategy.COSINE
)

# Print the number of documents in the database
print(db.index.ntotal)

# Save the database locally
db.save_local("faiss_index")