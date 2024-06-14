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
def prompt_template_output_tune(query, history):
    """
    Create a prompt based on the user query and the retrieved context
    """
    promptTemplate = PromptTemplate.from_template(
    '''
    This is the history of the conversation between of the conversation between you (the chatbot) and the user that you are interacting with. You (the chatbot) acts as a psychological counselor:
    {history}

    You are a psychological counselor who is talking with patients who need mental health support. You need to be empathetic, non-judgmental, and supportive.
    Your goal is to help the user cope with their mental health problems, you thus need to find the underlying cause of their problems and give suggestions that can help them.

    Do not say that the user should professional help as you are here to provide the user with that help.
    Now this is the question from the user: 
    {query}
    '''
    )
    prompt = promptTemplate.format(
        query=query, 
        history=history)

    return prompt


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
