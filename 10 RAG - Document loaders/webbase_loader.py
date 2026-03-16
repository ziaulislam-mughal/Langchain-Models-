from langchain_community.document_loaders import WebBaseLoader
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

# 5. Web Loader
url = "https://uajk.edu.pk/"
loader = WebBaseLoader(url)
docs = loader.load()

# 6. Create a Prompt Template
prompt = PromptTemplate(
    template = "Answer the following question \n {question} from the following text - \n {text}",
    input_variables = ["question", "text"]
)

# 7. Output Parser

parser = StrOutputParser()

# 8. Create a Chain

chain = prompt | model | parser
print(chain.invoke({
    "question": "What is the name of the university?",
    "text": docs[0].page_content
}))