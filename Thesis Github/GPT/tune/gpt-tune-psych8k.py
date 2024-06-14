import os
from openai import OpenAI
import json
import time
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()

# ACCESS OPENAI API
client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY']
)

train_data = '/Users/noahvanrijn/python-repos/master/Thesis/datasets/prepared_datasets/psych8k_train.jsonl'
val_data = '/Users/noahvanrijn/python-repos/master/Thesis/datasets/prepared_datasets/psych8k_val.jsonl'

# Upload the datasets to the OpenAI developer account
training_info = client.files.create(
  file=open(train_data, "rb"),
  purpose="fine-tune"
)

validation_id = client.files.create(
  file=open(val_data, "rb"),
  purpose="fine-tune"
)

print(f"Training File ID: {training_info}")
print(f"Validation File ID: {validation_id}")

training_id = training_info.id
validation_id = validation_id.id


# Create a fine-tuning job
jobs = client.fine_tuning.jobs.create(
  training_file=training_id, 
  validation_file=validation_id,
  model="gpt-3.5-turbo-0125",
  hyperparameters={
    "n_epochs":4
    # "batch_size":2,
    # "learning_rate_multiplier":8
  }
)

job_id = jobs.id

while True:
    job_status = client.fine_tuning.jobs.retrieve(job_id)
    if job_status.status in ['succeeded', 'failed']:
        break
    print(f"Current job status: {job_status.status}")
    time.sleep(10)  # Check the status every 10 seconds to avoid hitting the API rate limit


