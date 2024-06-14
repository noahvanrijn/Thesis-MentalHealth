# OpenAI key from environment
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import FAISS
import cohere
from dotenv import load_dotenv
from nltk.tokenize import word_tokenize
import pandas as pd

import json
import os


from helper_functions import prompt_template_output_RAG, prompt_template_HyDe, print_rerank_results, list_of_subqueries
#-----------------------------ENVIRONMENT VARIABLES----------------------------
# Load the environment variables
load_dotenv()

# Retrieve the Cohere API key from environment variables
cohere_api_key = os.getenv('COHERE_API_KEY')

# Initialize the Cohere client with the API key
co = cohere.Client(cohere_api_key)

#-----------------------------DATA INGESTION-----------------------------------
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

# Initialize the OpenAIEmbeddings class
embedding_model = OpenAIEmbeddings()

# ----------------------------ENSEMBLE RETRIEVER-------------------------------
# initialize the bm25 retriever and faiss retriever
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 4

# Load the vector database
db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

# Initialize the faiss retriever
faiss_retriever = db.as_retriever(search_kwargs={"k": 4})

# initialize the ensemble retriever
ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])

#---------------------------CREATE CONVERSATION CHAIN-------------------------
# Initialize the model parameters
#model = "ft:gpt-3.5-turbo-0125:personal::9SKdy2CT"
model = "gpt-3.5-turbo-0125"
max_tokens = 750
temperature = 0.2

# Initialize the ChatOpenAI class with the desired parameters
llm = ChatOpenAI(temperature=temperature, model_name=model, max_tokens=max_tokens)


def chatbot_response(user_query):
    #--------------------SUBQUERIES-----------------------------------
    # Split the user query into subqueries
    queries = list_of_subqueries(user_query, num_queries=3)

    print("Subqueries:", queries)

    # Initialize a list of the final documents
    final_docs = []

    # Perform the retrieval and reranking on the subqueries
    for query in queries:
        print("Query:", query)
        #-------------------------RETRIEVAL AND RERANKING-----------------------------------
        # Get the relevant documents for the user query
        docs_query = ensemble_retriever.get_relevant_documents(query)

        # Store the page content of the all retrieved documents from the user query as a list
        all_docs_query = [doc.page_content for doc in docs_query]

        # Perform reranking on the user query
        rerank_results_query = co.rerank(query=query, documents=all_docs_query, top_n=2, model='rerank-english-v2.0', return_documents=True)

        # Print the rerank results
        final_query_docs = print_rerank_results(rerank_results_query)

        # Concatenate the final documents from the user query
        final_docs.extend(final_query_docs)

        print("Final docs:", final_docs)

    #---------------------HYDE----------------------------------
    # Get the hypothetical response from the chatbot
    prompt = prompt_template_HyDe(user_query)
    response = llm.invoke(prompt)
    hypothetical_response = response.content

    #-------------------------RETRIEVAL AND RERANKING-----------------------------------
    # Get the relevant documents for the hypothetical response
    docs_hyde = ensemble_retriever.get_relevant_documents(hypothetical_response)

    # Store the page content of the all retrieved documents from the user query as a list
    all_docs_hyde = [doc.page_content for doc in docs_hyde]

    # Perform reranking on the user query
    rerank_results_hyde = co.rerank(query=hypothetical_response, documents=all_docs_hyde, top_n=2, model='rerank-english-v2.0', return_documents=True)

    # Print the rerank results
    final_hyde_docs = print_rerank_results(rerank_results_hyde)

    #-----------------------FINAL OUTPUT-------------------------------------
    # Concatenate the final documents from the user query and hypothetical response
    final_docs.extend(final_hyde_docs)

    print("Final docs:", final_docs)

    # Create a prompt using the user input and retrieved documents to give to the model
    prompt = prompt_template_output_RAG(user_query, final_docs)

    # Generate the chatbot's response
    answer = str(llm.invoke(prompt).content)
    
    return answer

# Load the questions dataframe
questions = pd.read_csv('/Users/noahvanrijn/python-repos/master/Thesis/datasets/output_datasets/counseling_questions_and_answers.csv')

# Apply function to DataFrame
questions['response_RAG'] = questions['questionText'].apply(chatbot_response)

# Save the updated DataFrame
questions.to_csv('/Users/noahvanrijn/python-repos/master/Thesis/datasets/output_datasets/counseling_questions_and_answers.csv', index=False)

