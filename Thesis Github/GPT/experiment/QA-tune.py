from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import pandas as pd

#from helper_functions import prompt_template_output_tune, create_memory, conversationsummarybuffer
from helper_functions import prompt_template_output_tune


#-----------------------------ENVIRONMENT VARIABLES----------------------------
# Load the environment variables
load_dotenv()
#---------------------------CREATE CONVERSATION CHAIN-------------------------
# Initialize the model parameters
# Change to fine-tuned model
model = "ft:gpt-3.5-turbo-0125:personal::9SKdy2CT"
max_tokens = 750
temperature = 0.2

# Initialize the ChatOpenAI class with the desired parameters
llm = ChatOpenAI(temperature=temperature, model_name=model, max_tokens=max_tokens)

#-----------------------------INTERACTIVE CHAT--------------------------------
def chatbot_response(user_query):
    # Create a prompt using the user input and retrieved documents to give to the model
    prompt = prompt_template_output_tune(user_query)

    # Generate the chatbot's response
    answer = str(llm.invoke(prompt).content)
    
    return answer

questions = pd.read_csv('/Users/noahvanrijn/python-repos/master/Thesis/datasets/prepared_datasets/counseling_questions_filtered.csv')

# Apply the function to each row in the dataframe and store the responses
questions['response_tune'] = questions['questionText'].apply(chatbot_response)

# Save the updated dataframe
questions.to_csv('/Users/noahvanrijn/python-repos/master/Thesis/datasets/output_datasets/counseling_questions_and_answers.csv', index=False)