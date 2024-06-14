import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv('step_metrics.csv')

print(df.head())

# plot every 10th step
step_interval = 10
df = df.iloc[(step_interval-1)::step_interval]

plt.plot(df['train_loss'], label='Training Loss')
plt.plot(df['valid_loss'], label='Validation Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()