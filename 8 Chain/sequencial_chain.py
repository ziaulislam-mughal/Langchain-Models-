import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

#chain used with ootput parser . using chain we can make a proper pipline . 

# 1. Load Environment
load_dotenv()

# 2. Check Token
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not token:
    print("Error: HUGGINGFACEHUB_API_TOKEN is not set in the .env file.")

# 3. Setup the Endpoint
repo_id = "meta-llama/Llama-3.1-8B-Instruct"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7,
    huggingfacehub_api_token=token,
)

# 4. Wrap it in ChatHuggingFace (This handles the "conversational" format for you)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser() 

template = PromptTemplate(
    template = """ give me five facts about {topic} """,
    input_variables = ["topic"]
)

prompt = template.format(topic="Islamic history")

chain = template | model | parser

response = chain.invoke({'topic': "Islamic history"})
print(response)