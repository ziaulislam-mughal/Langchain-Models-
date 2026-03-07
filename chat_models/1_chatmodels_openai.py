from langchain import OpenAIChat
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAIChat LLM

model = OpenAIChat(model="gpt-3.5-turbo")

# Example usage

results = model.invoke("What is the capital of France?")
print(results)  
# Output: 
print(results.content)