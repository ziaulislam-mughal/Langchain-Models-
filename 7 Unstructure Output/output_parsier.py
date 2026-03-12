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


# 1st prompt : detail report 
prompt_template_1 = PromptTemplate.from_template(
    "write a detail note on {topic}"
)

prompt_template_2 = PromptTemplate.from_template(
    "write a summary of five line following text {text}"
)


parser = StrOutputParser()


chain = prompt_template_1 | model | parser | prompt_template_2 | model | parser

result = chain.invoke({"topic": "Artificial Intelligence"})
print(result)