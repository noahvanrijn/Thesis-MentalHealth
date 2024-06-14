# Description: This file contains helper functions that are used in the RAG model
# Purpose: To define helper functions that are used in the RAG model
# IMPORTS
from langchain.prompts import PromptTemplate
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# Initialize the model parameters
model = "gpt-3.5-turbo-0125"
max_tokens = 500
temperature = 0.2

# Initialize the ChatOpenAI class with the desired parameters
llm = ChatOpenAI(temperature=temperature, model_name=model, max_tokens=max_tokens)

# Define a function to prompt the llm with the query and retrieved documents
def prompt_template_output_tune(query):
    """
    Create a prompt based on the user query and the retrieved context
    """
    promptTemplate = PromptTemplate.from_template(
    '''
    You are a psychological counselor who is talking with patients who need mental health support. You need to be empathetic, non-judgmental, and supportive.
    Your goal is to help the user cope with their mental health problems, you thus need to find the underlying cause of their problems and give suggestions that can help them.

    Do not say that the user should professional help as you are here to provide the user with that help.
    Now this is the question from the user:

    Question: {query}
    '''
    )
    prompt = promptTemplate.format(
        query=query
        )

    return prompt

# Define a function to prompt the llm with the query and retrieved documents
def prompt_template_HyDe(query):
    """
    Create a prompt based on the user query and the retrieved context
    """
    promptTemplate = PromptTemplate.from_template(
    '''
    You are a psychological counselor that is talking with patients that need mental health support. You need to be empthatic, non-judgmental and supportive.
    The user will ask you questions about mental health and you need to provide them with the best possible answers.
    Use a maximum of 256 tokens to answer the user's question.

    Now this is the question from the user:

    Question: {query}
    '''
    )
    prompt = promptTemplate.format(
        query=query)

    return prompt


def main_issues(query):
    """
    Extract the main issues from the user query
    """
    promptTemplate = PromptTemplate.from_template(
    '''
    You will get a user query that is related to mental health. 
    The goal is to retrieve the main issues from the user query so these can be used for the semantic retrieval of a retrieval augmented-generation system. 
    Ensure that the main issues capture as much information as possible. your answer should contain only the main issues separated by a “\n” and nothing else. 
    Furthermore the words “treatment for” should be included before each main issue, an example input and output is given below.

    For example this query:
    "I am a teenager. I have been experiencing major episodes of depression (if that's even what it is) for several years. It’s always getting worse. I have been having panic attacks, feeling like I can't control my fears, and I can't even bring myself to care if I live or die anymore. The problem is that I don't know who to ask for help. When I try to talk to my parents, I freeze completely and can't do anything but make a joke because never once in my life have we talked about our feelings.”

    Should have this output:

    "Treatment for teenagers experiencing severe depression that is getting worse, panic attacks accompanied by intense fears and apathy about living.
    Treatment for challenges in communicating feelings to their parents, resulting in avoidance of serious discussions about mental health."
    
    There could be less or more main issues than in the example, include as many as possible. Now do the same for this user query: {query}
    
    '''
    )
    prompt = promptTemplate.format(
        query=query)

    return prompt


def prompt_template_rewrite_query(query, num_queries):
    """
    Create a prompt based on the user query and the retrieved context
    """
    promptTemplate = PromptTemplate.from_template(
    '''
    You are a helpful assistant that generates multiple search queries based on a 
    single input query. Generate {num_queries} search queries, one on each line, 
    related to the following input query:
    Query: {query}
    Queries:
    '''
    )
    prompt = promptTemplate.format(
        query=query,
        num_queries=num_queries)

    return prompt


def list_of_subqueries(query, num_queries):
    prompt = prompt_template_rewrite_query(query, num_queries)
    response = llm.invoke(prompt).content
    queries = response.split("\n")
    return queries 


# NOTE - read about prompts and how to formulate the perfect prompt for this problem
# Define a function to prompt the llm with the query and retrieved documents
def prompt_template_output_RAG(query, context):
    """
    Create a prompt based on the user query and the retrieved context
    """
    promptTemplate = PromptTemplate.from_template(
    '''
    You are a psychological counselor who is talking with patients who need mental health support. You need to be empathetic, non-judgmental, and supportive.
    Your goal is to help the user cope with their mental health problems, you thus need to find the underlying cause of their problems and give suggestions that can help them.
    
    For the questions you can use the information provided below, that can be used as a knowledge base for you to provide better answers.
    This is the information you can use:
    {context}

    Do not use to many tokens and keep the conversation natural.

    Do not say that the user should professional help as you are here to provide the user with that help.
    Now this is the question from the user:

    Question: {query}
    '''
    )
    prompt = promptTemplate.format(
        query=query, 
        context=context
        )

    return prompt

def print_rerank_results(rerank_results):
    list_query_docs = []

    for idx, r in enumerate(rerank_results.results):
        print(f"Document Rank: {idx + 1}, Document Index: {r.index}")
        print(f"Document: {r.document.text}")
        print(f"Relevance Score: {r.relevance_score:.2f}")
        print("\n")
        if r.relevance_score > 0.8:
            list_query_docs.append(r.document.text)

    return list_query_docs


def reformulate_query_to_256_tokens(query):
    """
    Reformulate the query to a maximum of 256 tokens using the openai API
    """
    promptTemplate = PromptTemplate.from_template(
    '''
    Can you reformulate the question from the user to a maximum of 256 tokens? 
    It is important to keep the meaning of the question intact.

    Now this is the question from the user:

    Question: {query}
    '''
    )
    prompt = promptTemplate.format(
        query=query)

    return prompt

def split_string_whitespace(input_string):
    """
    Split the output from the llm into the main topics
    """
    lines = input_string.strip().split('\n')

    return lines
    

def convert_string_to_lists(input_string):
    """
    Convert a string containing lists to actual list objects
    """
    # Creating an empty dictionary to safely execute the strings within a controlled environment
    local_vars = {}

    # Splitting the input string into individual lines and stripping any whitespace
    lines = input_string.strip().split('\n')

    # Executing each line using exec() within the safe local environment
    for line in lines:
        exec(line, {}, local_vars)

    # Extracting the variables from the local_vars dictionary
    main_issues_list = local_vars['main_issues']
    emotional_states_list = local_vars['emotional_states']
    type_support_list = local_vars['type_support']

    return main_issues_list, emotional_states_list, type_support_list


def create_memory(user_query, answer):
    """
    Create a memory of the user query and the chatbot answer
    """
    memory_turn = {
        "user_query": user_query,
        "chatbot_answer": answer
    }
    return memory_turn


def prompt_template_summary(query):
    """
    Create a prompt based on the user query and the retrieved context
    """
    promptTemplate = PromptTemplate.from_template(
    '''
    Can you summarize this conversation between a user and a chatbot that acts as a psychological counselor?
    Use a maximum of 350 tokens, you can use less if necessary

    conversation: {query}

    '''
    )
    prompt = promptTemplate.format(
        query=query)

    return prompt


def conversationsummarybuffer(memory, summary_threshold):
    """
    Summarize parts of the conversation memory that exceed a token length threshold while retaining other parts.
    """
    # Concatenate all user queries and answers into one large string to check total token count
    full_text = " ".join([turn["user_query"] + " " + turn["chatbot_answer"] for turn in memory])
    tokens = word_tokenize(full_text)

    memory_and_summary = memory
    
    if len(tokens) > summary_threshold:
        # If tokens exceed the threshold, find the point where the threshold is surpassed
        cumulative_tokens = 0
        for index, turn in enumerate(memory):
            turn_text = turn["user_query"] + " " + turn["chatbot_answer"]
            turn_tokens = word_tokenize(turn_text)
            cumulative_tokens += len(turn_tokens)
            if cumulative_tokens > summary_threshold:
                break
        
        # Make a disctionton between the working memory and the old memory
        working_memory = memory[:index]
        old_memory = memory[index:]
        
        # Summarize the conversation from the old memory
        text_to_summarize = " ".join([turn["user_query"] + " " + turn["chatbot_answer"] for turn in old_memory])
        prompt = prompt_template_summary(text_to_summarize)
        summary = str(llm.invoke(prompt).content)

        query_summary = '"Give a summary of the conversation that has already happend before"'

        # Replace the original memory with the summary starting from the index where the threshold was exceeded
        memory_and_summary = working_memory + [{"user_query": query_summary, "chatbot_answer": summary}] 
    
    return memory_and_summary


def check_256_tokens(user_query):
    """
    Check if the query has more than 256 tokens
    If it has more than 256 tokens, reforumulate the user query to a maximum of 256 tokens
    Otherwise return the original query
    """
    tokens = word_tokenize(user_query)
    if len(tokens) > 256:
        prompt_256_tokens = reformulate_query_to_256_tokens(user_query)
        user_query_256_tokens = llm.invoke(prompt_256_tokens).content
        return user_query_256_tokens
    else:
        return user_query