from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

#from helper_functions import prompt_template_output_tune, create_memory, conversationsummarybuffer
from helper_functions import prompt_template_output_tune, create_memory, conversationsummarybuffer


#-----------------------------ENVIRONMENT VARIABLES----------------------------
# Load the environment variables
load_dotenv()
#---------------------------CREATE CONVERSATION CHAIN-------------------------
# Initialize the model parameters
# Change to fine-tuned model
model = "ft:gpt-3.5-turbo-0125:personal::9SKdy2CT"
max_tokens = 750
temperature = 0.2

summary_threshold = 4000

# Initialize the ChatOpenAI class with the desired parameters
llm = ChatOpenAI(temperature=temperature, model_name=model, max_tokens=max_tokens)

#-----------------------------INTERACTIVE CHAT--------------------------------
def chatbot_response(user_query, memory, memory_and_summary):
    # Create a prompt using the user input and retrieved documents to give to the model
    prompt = prompt_template_output_tune(user_query, memory_and_summary)

    # Generate the chatbot's response
    answer = str(llm.invoke(prompt).content)

    # Print the chatbot's response
    print("Chatbot:", answer)
    
    # Create a memory of the user query and the chatbot answer
    memory_turn = create_memory(user_query, answer)
    memory.insert(0, memory_turn)

    # Create memory and a summary of older questions and answers based on threshold
    memory_and_summary = conversationsummarybuffer(memory, summary_threshold)

    print("Memory:", memory)
    print("Memory and summary:", memory_and_summary)
    
    return answer, memory, memory_and_summary

        